from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import StateGraph, END

from newsroomagent.models import NewsroomState
from newsroomagent.providers import get_chat_model


# MCP CLIENT CONFIG. TELLS ADAPTER HOW TO LAUNCH THE SERVER.
MCP_CONFIG = {
    "newsroomagent": {
        "command": "uv",
        "args": ["run", "python", "-m", "newsroomagent.mcp_server"],
        "transport": "stdio",
    }
}


def filter_tools(tools, names):
    return [t for t in tools if t.name in names]

#RESEARCHER
RESEARCHER_PROMPT = """You are a news researcher gathering facts for a news segment.
Use available tools to investigate the topic:
Use archive_search first. It contains details on world events from the last month or so.
Use web_search only if the topic is breaking news or not found in the archive.
Use get_current_time if the question depends on what day 'today' is.
Return a brief summary of your findings as a single text block.
Include source filenames or URLs inline (e.g. "[2026-bulgarian-parliamentary-election.txt]")
so a fact-checker can verify claims later."""

def make_researcher_node(tools):
    # FILTER TO ONLY THE TOOLS THE RESEARCHER NEEDS.
    research_tools = filter_tools(
        tools, ["archive_search", "web_search", "get_current_time"]
    )

    # BUILD THE LLM AND THE AGENT ONCE
    llm = get_chat_model(temperature=0)
    agent = create_agent(llm, research_tools, system_prompt=RESEARCHER_PROMPT)

    async def researcher_node(state: NewsroomState) -> dict:
        # START AGENT LOOP WITH. FEED USER'S PROMPT
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"Research this topic: {state['topic']}")]
        })
        # THE LAST MESSAGE IS THE LLM'S FINAL TEXT REPLY AFTER ALL TOOL CALLS FINISHED.
        notes = result["messages"][-1].content
        return {"research_notes": notes}

    return researcher_node

# FACT CHECKER
class ClaimVerdict(BaseModel):
    """SINGLE FACT-CHECK VERDICT FOR A SINGLE CLAIM."""
    # FIELD DESCRIPTIONS ARE READ BY LLM WHEN DECIDING WHAT TO PUT INTO EACH VARIABLE
    claim: str = Field(description="The factual claim being verified")
    verified: bool = Field(description="True if the archive supports the claim, False if contradicted or unsupported")
    evidence: str = Field(description="Quoted or paraphrased archive content used to make the verdict")


class FactCheckReport(BaseModel):
    """COLLECTION OF VERDICTS"""
    verdicts: list[ClaimVerdict]

FACTCHECKER_PROMPT = """You are a fact-checker for a news segment.
You will receive a researcher's notes. Your job:
1. Identify the key factual claims (dates, numbers, names, events, results).
2. Use the archive_search tool to look up source material for each claim.
3. Decide if the archive supports or contradicts each claim.
When done, write a clear analysis listing each claim with your verdict
and the supporting evidence you found. Focus on claims that affect the
news story. Skip trivial details."""

FORMATTER_PROMPT = """Convert the following fact-check analysis into a structured list of verdicts.
Analysis:
{analysis}
Return one ClaimVerdict per checked claim. Use the exact claim text from the analysis."""

def make_factchecker_node(tools):
    """FACTCHECKER NODE. VERIFIES CLAIMS AGAINST THE ARCHIVE."""
    # FACTCHECKER WILL ONLY USE archive_search TOOL
    fc_tools = filter_tools(tools, ["archive_search"])

    llm = get_chat_model(temperature=0)
    agent = create_agent(llm, fc_tools, system_prompt=FACTCHECKER_PROMPT)

    # SECOND LLM RETURNS A FactCheckReport INSTANCE
    formatter_llm = get_chat_model(temperature=0).with_structured_output(FactCheckReport)

    async def factchecker_node(state: NewsroomState) -> dict:
        # RUNS AGENT
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=(
                f"Topic: {state['topic']}\n\n"
                f"Research notes to fact-check:\n{state['research_notes']}"
            ))]
        })
        analysis = result["messages"][-1].content

        # RUNS FORMATTER
        report = await formatter_llm.ainvoke(
            FORMATTER_PROMPT.format(analysis=analysis)
        )

        # CONVERT PYDANTIC OBJECT INTO PLAIN DICT
        verdicts = [v.model_dump() for v in report.verdicts]
        return {"fact_check_results": verdicts}

    return factchecker_node


# WRITER
WRITER_PROMPT = """You are a news writer creating a news story segment.
You will receive the topic from the user, esearch notes from a researcher and
fact checked verdicts on the key claims.

Your job is to write a brief and clear news script (about 400 words max) 
covering the verified information.

RULES:
1. Use ONLY claims marked VERIFIED in the fact-check report.
2. If a key fact was REJECTED, skip it.
3. Write in news-anchor style: neutral and factual.
4. No filler, no editorializing, no "in conclusion."
5. End with a one-line attribution: "Source: NewsroomAgent archive."

Structure: a lede paragraph with the most important fact, then 2-4 supporting paragraphs."""


def make_writer_node():
    llm = get_chat_model(temperature=0.3)

    async def writer_node(state: NewsroomState) -> dict:
        # FORMAT FACT-CHECK RESULTS AS A LABELED LIST FOR THE LLM TO REFERENCE.
        fc_lines = []
        for v in state.get("fact_check_results", []):
            tag = "VERIFIED" if v["verified"] else "REJECTED"
            fc_lines.append(f"[{tag}] {v['claim']}")
        fc_summary = "\n".join(fc_lines) if fc_lines else "(no fact-check available)"

        user_msg = (
            f"Topic: {state['topic']}\n\n"
            f"Research notes:\n{state['research_notes']}\n\n"
            f"Fact-check verdicts:\n{fc_summary}\n\n"
            "Write the news segment script now."
        )

        response = await llm.ainvoke([
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(content=user_msg),
        ])
        return {"draft_script": response.content}

    return writer_node

# SUPERVISOR/ ROUTER
MAX_STEPS = 6

class SupervisorRouter(BaseModel):
    next: Literal["researcher", "factchecker", "writer", "FINISH"] = Field(
        description="The next node to execute, or FINISH to stop."
    )
    reason: str = Field(description="One short sentence explaining the choice.")

SUPERVISOR_PROMPT = """You are the supervisor of a 3 agent news-research team.
Your team:
    researcher: gathers facts using archive_search, web_search, get_current_time tools.
    factchecker: verifies the researcher's claims against the archive.
    writer: produces the final news script from verified research.

Routing rules:
1. If research_notes is empty, route to researcher.
2. If research_notes exists but fact-check results are empty, route to factchecker.
3. If more than 10% of claims were REJECTED, route back to researcher for another pass.
4. If most claims are VERIFIED, route to writer.
5. Choose FINISH after the writer has produced the draft_script.

# STATE SNAPSHOT
Current state:
- topic: {topic}
- has research_notes: {has_notes}
- fact-check results: {fc_count} total ({verified_count} verified, {rejected_count} rejected)
- has draft_script: {has_draft}
- step_count: {step_count} / {max_steps}
"""

# SUPERVISOR NODE. ROUTES BETWEEN AGENTS BASED ON STATE
def make_supervisor_node():
    llm = get_chat_model(temperature=0).with_structured_output(SupervisorRouter)

    async def supervisor_node(state: NewsroomState) -> dict:
        step = state.get("step_count", 0) + 1

        # STEP BUDGET GUARD. RUN BEFORE INVOKING LLM
        if step >= MAX_STEPS:
            if state.get("research_notes"):
                print(f"  [supervisor step {step}] BUDGET EXCEEDED, FORCING WRITER")
                return {"next": "writer", "step_count": step}
            print(f"  [supervisor step {step}] BUDGET EXCEEDED WITH NO NOTES, FORCING FINISH")
            return {"next": "FINISH", "step_count": step}

        fc_results = state.get("fact_check_results", []) or []
        verified = sum(v["verified"] for v in fc_results)
        rejected = len(fc_results) - verified

        prompt = SUPERVISOR_PROMPT.format(
            topic=state.get("topic", ""),
            has_notes=bool(state.get("research_notes")),
            fc_count=len(fc_results),
            verified_count=verified,
            rejected_count=rejected,
            has_draft=bool(state.get("draft_script")),
            step_count=step,
            max_steps=MAX_STEPS,
        )

        decision = await llm.ainvoke(prompt)
        print(f"  [supervisor step {step}] -> {decision.next} ({decision.reason})")

        return {"next": decision.next, "step_count": step}

    return supervisor_node

# CONDITIONAL EDGE: READS state['next'] AND RETURNS A NODE NAME OR END
def route_from_supervisor(state: NewsroomState) -> str:
    next_step = state["next"]
    if next_step == "FINISH":
        return END
    return next_step

# ASSEMBLE MULTI-AGENT GRAPH. RETURNS A COMPILED GRAPH
def build_graph(tools):
    g = StateGraph(NewsroomState)

    # REGISTER ALL NODES
    g.add_node("supervisor", make_supervisor_node())
    g.add_node("researcher", make_researcher_node(tools))
    g.add_node("factchecker", make_factchecker_node(tools))
    g.add_node("writer", make_writer_node())

    g.set_entry_point("supervisor")

    # SUPERVISOR ROUTES TO ONE OF FOUR DESTINATIONS
    g.add_conditional_edges(
        "supervisor", route_from_supervisor,
        {
            "researcher": "researcher",
            "factchecker": "factchecker",
            "writer": "writer",
            END: END,
        }
    )

    # LOOP BACK TO SUPERVISOR; WRITER WILL END.
    g.add_edge("researcher", "supervisor")
    g.add_edge("factchecker", "supervisor")
    g.add_edge("writer", END)

    return g.compile()


# SMOKE TEST FOR MCP TOOL DISCOVERY
if __name__ == "__main__":
    async def smoke():
        client = MultiServerMCPClient(MCP_CONFIG)
        # PERSISTENT SESSION KEEPS ONE MCP SUBPROCESS ALIVE FOR THE WHOLE GRAPH RUN
        async with client.session("newsroomagent") as session:

            tools = await load_mcp_tools(session)
            print(f"LOADED {len(tools)} TOOLS FROM MCP SERVER.")

            topic = "What elections happened in India in 2026?"
            graph = build_graph(tools)
            initial_state = {
                "topic": topic,
                "research_notes": "",
                "fact_check_results": [],
                "draft_script": "",
                "next": "",
                "step_count": 0,
            }

            print(f"INVOKING GRAPH ON: {topic}\n")
            final_state = await graph.ainvoke(initial_state)

            print("\n")
            print(f"SUMMARY: {final_state['step_count']} supervisor turns")
            print("\n")
            verified = sum(1 for v in final_state["fact_check_results"] if v["verified"])
            rejected = len(final_state["fact_check_results"]) - verified
            print(f"fact-check: {verified} verified, {rejected} rejected")

            print("\n")
            print("FINAL NEWS SCRIPT")
            print("\n")
            print(final_state["draft_script"])

    asyncio.run(smoke())