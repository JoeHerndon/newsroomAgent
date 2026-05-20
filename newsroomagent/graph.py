import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from pydantic import BaseModel, Field

from newsroomagent.config import CHAT_MODEL
from newsroomagent.models import NewsroomState


# MCP CLIENT CONFIG. TELLS ADAPTER HOW TO LAUNCH THE SERVER.
MCP_CONFIG = {
    "newsroomagent": {
        "command": "uv",
        "args": ["run", "python", "-m", "newsroomagent.mcp_server"],
        "transport": "stdio",
    }
}

async def load_tools():
    """SPAWN THE MCP SERVER AND RETURN ITS TOOLS AS LANGCHAIN BaseTool OBJECTS."""
    client = MultiServerMCPClient(MCP_CONFIG)
    tools = await client.get_tools()
    return tools

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
    llm = ChatAnthropic(model=CHAT_MODEL, temperature=0)
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

    llm = ChatAnthropic(model=CHAT_MODEL, temperature=0)
    agent = create_agent(llm, fc_tools, system_prompt=FACTCHECKER_PROMPT)

    # SECOND LLM RETURNS A FactCheckReport INSTANCE
    formatter_llm = ChatAnthropic(model=CHAT_MODEL, temperature=0).with_structured_output(FactCheckReport)

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
    llm = ChatAnthropic(model=CHAT_MODEL, temperature=0.3)

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


# SMOKE TEST FOR MCP TOOL DISCOVERY
if __name__ == "__main__":
    async def smoke():
        tools = await load_tools()
        print(f"LOADED {len(tools)} TOOLS FROM MCP SERVER.")

        topic = "What elections happened in India in 2026?"
        state = {"topic": topic}

        # RUN RESEARCHER
        researcher = make_researcher_node(tools)
        state.update(await researcher(state))
        print("RESEARCH DONE.\n")
        # print(state["research_notes"])

        # RUN FACTCHECKER
        factchecker = make_factchecker_node(tools)
        state.update(await factchecker(state))
        verified_count = sum(1 for v in state["fact_check_results"] if v["verified"])
        print(f"FACT CHECKER DONE. {verified_count}/{len(state['fact_check_results'])} VERIFIED.\n")
        # for v in state["fact_check_results"]:
        #     tag = "[VERIFIED]" if v["verified"] else "[REJECTED]"
        #     print(f"\n{tag} {v['claim']}")
        #     print(f"  EVIDENCE: {v['evidence']}")

        # RUN WRITER
        writer = make_writer_node()
        state.update(await writer(state))
        print("WRITER DONE.\n")

        print("--- FINAL NEWS SCRIPT ---\n")
        print(state["draft_script"])

    asyncio.run(smoke())