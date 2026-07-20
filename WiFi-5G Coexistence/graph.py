from langgraph.graph import StateGraph, START, END
from state import GraphState
from agents import wifi_agent_node, nr_u_agent_node
from validator import safety_validator_node

# Maximum allowed negotiation iterations before forcing a fallback
MAX_ITERATIONS = 3

def fallback_node(state: GraphState) -> dict:
    """
    Deterministic safety fallback executed if agents get stuck in 
    a negotiation loop or propose physically invalid parameters.
    """
    return {
        "final_configuration": {
            "allocation_wifi_ms": 5.0,
            "allocation_nru_ms": 5.0,
            "icc_agreed": True # Safe default when under potential congestion
        },
        "status": "fallback_applied"
    }

# Router logic after 5G Agent evaluates Wi-Fi proposal
def route_after_nru(state: GraphState) -> str:
    status = state.get("status")
    iteration = state.get("iteration_count", 0)
    
    # 1. Guardrail against Infinite Loop DoS Attacks
    if iteration >= MAX_ITERATIONS:
        print(f"\n[Guardrail Triggered]: Max iterations ({MAX_ITERATIONS}) reached. Routing to Fallback.")
        return "fallback"
        
    # 2. Both agents reached a compromise
    if status == "AGREED":
        return "validator"
        
    # 3. Counter-offer made; continue negotiation loop
    return "wifi_agent"

# Router logic after Safety Validator runs
def route_after_validator(state: GraphState) -> str:
    status = state.get("status")
    if status == "accepted":
        return END
    return "fallback"

# Build the LangGraph Workflow
builder = StateGraph(GraphState)

# Add Nodes
builder.add_node("wifi_agent", wifi_agent_node)
builder.add_node("nr_u_agent", nr_u_agent_node)
builder.add_node("validator", safety_validator_node)
builder.add_node("fallback", fallback_node)

# Define Edges & Flow
builder.add_edge(START, "wifi_agent")
builder.add_edge("wifi_agent", "nr_u_agent")

builder.add_conditional_edges(
    "nr_u_agent",
    route_after_nru,
    {
        "validator": "validator",
        "wifi_agent": "wifi_agent",
        "fallback": "fallback"
    }
)

builder.add_conditional_edges(
    "validator",
    route_after_validator,
    {
        END: END,
        "fallback": "fallback"
    }
)

builder.add_edge("fallback", END)

# Compile Graph
app = builder.compile()