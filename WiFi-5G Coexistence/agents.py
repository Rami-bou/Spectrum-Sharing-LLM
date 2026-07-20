import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from state import GraphState
from prompts import WIFI_AGENT_SYSTEM_PROMPT, NR_U_AGENT_SYSTEM_PROMPT

llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)

def wifi_agent_node(state: GraphState) -> dict:
    """LangGraph node representing the Wi-Fi Access Point Agent."""
    telemetry = state.get("telemetry", {})
    history = state.get("negotiation_history", [])
    
    user_content = f"""
    CURRENT SEMANTIC TELEMETRY:
    {json.dumps(telemetry, indent=2)}

    NEGOTIATION HISTORY:
    {chr(10).join(history[-4:]) if history else "No history yet. Make initial proposal."}
    
    Provide your proposal in JSON as instructed in your system prompt.
    """
    
    messages = [
        SystemMessage(content=WIFI_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ]
    
    response = llm.invoke(messages)
    proposal = json.loads(response.content)
    
    new_history_entry = f"Wi-Fi Agent Proposal: {json.dumps(proposal)}"
    
    return {
        "wifi_latest_proposal": proposal,
        "negotiation_history": [new_history_entry],
        "iteration_count": state.get("iteration_count", 0) + 1,
        "status": proposal.get("negotiation_status", "PROPOSING")
    }

def nr_u_agent_node(state: GraphState) -> dict:
    """LangGraph node representing the 5G NR-U Base Station Agent."""
    telemetry = state.get("telemetry", {})
    wifi_prop = state.get("wifi_latest_proposal", {})
    history = state.get("negotiation_history", [])
    
    user_content = f"""
    CURRENT SEMANTIC TELEMETRY:
    {json.dumps(telemetry, indent=2)}

    LATEST WI-FI PROPOSAL:
    {json.dumps(wifi_prop, indent=2)}

    NEGOTIATION HISTORY:
    {chr(10).join(history[-4:]) if history else "Initial evaluation round."}
    
    Evaluate the proposal and respond in JSON as instructed in your system prompt.
    """
    
    messages = [
        SystemMessage(content=NR_U_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ]
    
    response = llm.invoke(messages)
    proposal = json.loads(response.content)
    
    new_history_entry = f"5G Agent Evaluation: {json.dumps(proposal)}"
    
    return {
        "nr_u_latest_proposal": proposal,
        "negotiation_history": [new_history_entry],
        "status": proposal.get("negotiation_status", "COUNTER_OFFER")
    }