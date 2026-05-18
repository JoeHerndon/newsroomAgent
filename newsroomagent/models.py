from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class NewsroomState(TypedDict):
    topic: str

    research_notes: str
    
    # FACTCHECKER OUTPUT. EACH ENTRY WILL HAVE A CLAIM, VERIFIED(bool), AND EVIDENCE
    fact_check_results: list[dict]

    draft_script: str

    # NEXT NODE (researcher | factchecker | writer | FINISH).
    next: str

    # INCREMENTS EACH TIME WE ENTER THE SUPERVISOR.
    step_count: int