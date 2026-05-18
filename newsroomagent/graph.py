import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

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

# DEFINE ROLE OF RESEARCHER
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


# SMOKE TEST FOR MCP TOOL DISCOVERY
if __name__ == "__main__":
    async def smoke():
        tools = await load_tools()
        print(f"LOADED {len(tools)} TOOLS FROM MCP SERVER.")

        researcher = make_researcher_node(tools)
        result = await researcher({"topic": "What elections happened in India in 2026?"})

        print("\n--- RESEARCH NOTES ---")
        print(result["research_notes"])

    asyncio.run(smoke())