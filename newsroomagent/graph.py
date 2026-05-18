import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


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


# SMOKE TEST FOR MCP TOOL DISCOVERY
if __name__ == "__main__":
    tools = asyncio.run(load_tools())
    print(f"LOADED {len(tools)} TOOLS FROM MCP SERVER:")
    for t in tools:
        print(f"  - {t.name}: {t.description[:100]}...")