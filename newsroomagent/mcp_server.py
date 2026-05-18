import sys,os
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from newsroomagent.retrieval import retrieve
from datetime import datetime, timezone
from tavily import TavilyClient

load_dotenv(Path(__file__).parent.parent / ".env")

# REGISTER SERVER
mcp = FastMCP("newsroomagent")

# ARCHIVE SEARCH TOOL TO CALL RETRIEVE FUNCTION
@mcp.tool()
def archive_search(query: str, k: int = 5) -> list[dict]:
    """Search the local news archive that covers world events from the past ~30 days
    for past coverage on a topic.
    Use this first for any news event question. The archive likely has coverage on most 
    recent internation incidents.
    Returns up to `k` chunks with source filename and excerpt.
    Do not use for topics outside the archive or for recent events of
    the last 24 hours"""

    chunks = retrieve(query, k=k)
    return [
        {"source": c.metadata.get("source", "unknown"),
        "excerpt": c.page_content[:400]}
        for c in chunks
    ]

@mcp.tool()
def get_current_time() -> str:
    """Return the current UTC timestamp in ISO 8601 format. Use this when
    the user asks 'what time is it', when computing date math, or when
    the answer depends on what 'today' is."""
    return datetime.now(timezone.utc).isoformat()

# WEB SEARCH VIA TAVILY. FOR BREAKING NEWS OR TOPICS OUTSIDE THE ARCHIVE
@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Use only when archive search returns no relative results.
    Search the web for news articles created within the last 24 hours on topics
    outside the local archive. Return results with title, 
    url, and snippet. Be specific and return the actual article/url.
    Do not use for historical/background context on older stories
    (use archive_search)"""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(query, max_results=max_results)
    return [{"title": r["title"], "url": r["url"], "snippet": r["content"]}
            for r in results["results"]]


if __name__ == "__main__":
    # STDERR ONLY
    print("STARTING NEWSROOMAGENT MCP SERVER", file=sys.stderr)  
    mcp.run(transport="stdio")


