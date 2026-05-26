import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from pathlib import Path

from newsroomagent.graph import MCP_CONFIG, build_graph


app = FastAPI(title="NewsroomAgent")
FRONTEND_DIR = Path(__file__).parent / "frontend"

# FRAME GENERATOR
async def run_stream(topic: str):
    # SESSION STAYS OPEN
    client = MultiServerMCPClient(MCP_CONFIG)
    async with client.session("newsroomagent") as session:
        tools = await load_mcp_tools(session)
        graph = build_graph(tools)

        initial_state = {
            "topic": topic,
            "research_notes": "",
            "fact_check_results": [],
            "draft_script": "",
            "next": "",
            "step_count": 0,
        }

        final_state = dict(initial_state)
        # STREAM THE GRAPH ONE NODE AT A TIME. EVERY TIME A NODE FINISHES IT MERGES
        # CHANGES INTO final_state AND PUSHES A SMALL SSE FRAME SO THE BROWSER SEES PROGRESS LIVE.
        async for event in graph.astream(initial_state, stream_mode="updates"):
            for node_name, delta in event.items():
                final_state.update(delta)
                payload = {
                    "node": node_name,
                    "step": final_state["step_count"],
                    "changes": preview_change(delta),
                }
                yield f"data: {json.dumps(payload)}\n\n"

        done = {"done": True, "script": final_state["draft_script"]}
        yield f"data: {json.dumps(done)}\n\n"

def preview_change(change):
    # SAME SHAPE AS THE CLI STREAM. WILL SHOW EVERY FIELD THE NODE CHANGED.
    preview = {}
    for k, v in change.items():
        if k == "fact_check_results":
            verified = sum(1 for x in v if x.get("verified"))
            preview[k] = f"{len(v)} claims, {verified} verified, {len(v) - verified} rejected"
        elif isinstance(v, str):
            preview[k] = v[:200] + "..." if len(v) > 200 else v
        elif isinstance(v, list):
            preview[k] = f"({len(v)} items)"
        else:
            preview[k] = v
    return preview


@app.get("/stream")
async def stream(topic: str):
    # SSE STREAM.
    return StreamingResponse(run_stream(topic), media_type="text/event-stream")

@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")