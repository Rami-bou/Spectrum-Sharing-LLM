from typing import List, Dict, Any, Optional, TypedDict, Literal, Tuple
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END 
from typing_extensions import TypedDict, Annotated

from primary_agent import primary
from secondary_agent import secondary

def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
    print("Finalizer...\n")
    if state["iteration"] > 3:
        return "finalize"
    # earsly stop
    if state['primary_decision'] == "ACCEPT":
        return "finalize"

    return "revise"

workflow = StateGraph(GraphState)

workflow.add_node("Primary", primary)
workflow.add_node("Secondary", secondary)

workflow.set_entry_point("Secondary")
workflow.add_edge("Secondary", "Primary")

workflow.add_conditional_edges(
    "Primary",
    finalizer,
    {
        "revise": "Secondary",
        "finalize": END,
    }
)

app = workflow.compile()