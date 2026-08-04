import matplotlib.pyplot as plt
import random
import numpy as np
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
import os
from langsmith import Client
from typing import List, Dict, Any, Optional, TypedDict, Literal, Tuple
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
import math

repeated = set()
data = []
f = 2e9
c = 3e8
wave = c / f
P_max = 100.0
I_max = 1000
scale_factor = 1e8
possible_P2 = []
P1 = 100
# random.seed(10)

def gen_channels():
  txp = [random.uniform(0, 10), random.uniform(26, 50)]
  txs = [random.uniform(0, 10), random.uniform(0, 25)]
  rxp = [random.uniform(15, 50), random.uniform(0, 25)]
  rxs = [random.uniform(15, 50), random.uniform(26, 50)]

  dpp = np.sqrt((txp[0]-rxp[0])**2 + (txp[1]-rxp[1])**2)
  dsp = np.sqrt((txs[0]-rxp[0])**2 + (txs[1]-rxp[1])**2)
  dps = np.sqrt((txp[0]-rxs[0])**2 + (txp[1]-rxs[1])**2)
  dss = np.sqrt((txs[0]-rxs[0])**2 + (txs[1]-rxs[1])**2)

  hpp = (wave / (4 * np.pi * dpp))**2
  hsp = (wave / (4 * np.pi * dsp))**2
  hps = (wave / (4 * np.pi * dps))**2
  hss = (wave / (4 * np.pi * dss))**2

  hpp_normal = int(round(hpp * scale_factor, 2))
  hsp_normal = int(round(hsp * scale_factor, 2))
  hps_normal = int(round(hps * scale_factor, 2))
  hss_normal = int(round(hss * scale_factor, 2))

  allowed_p2 = I_max / hsp_normal
  best_P2 = int(round(min(allowed_p2, P_max), 2))
  sinr = (P1 * hpp_normal) / (1 + best_P2 * hsp_normal)
  
  print(f"Generated Data Point: Hpp: {hpp_normal}, Hsp: {hsp_normal}, Hps: {hps_normal}, Hss: {hss_normal}, Best P2: {best_P2}")
  print(f"Interference: {best_P2*hsp_normal}")
  return hpp_normal, hsp_normal, hps_normal, hss_normal, best_P2

def calc_sinr(hpp, hsp, best_P2):
  return np.log2(1 + (P1 * hpp) / (1 + best_P2 * hsp))

def calc_interference(hsp, best_P2):
  return best_P2 * hsp

def allowed_gap(current_val, max_val):
  gaps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  if max_val - current_val in gaps:
    return True
  return False

class GraphState(TypedDict):
  P1: int
  P2: int
  hpp: int # Primary knows only its channel gain and transmission power
  hsp: int
  hss: int # Secondary knows its channel gain and allocate its own P2
  sinr: float
  caused_interference: int
  iteration: int
  max_iter: int
  accept: bool # whether the proposal accepted or not
  critique: Optional[List[str]] # P1 write a critic to P2, and P2 take it account

PROMPT_SECONDARY = """You are the secondary use in a wireless communication environment.
Your role is to set your transmission power based on you channel gain hss.
The allocation power is from 1 to 100, where 100 is the maximum value you can allocate.
You start with allocate the max power then you adjust based on the received critique.

Return JSON matching the schema.
"""

PROMPT_PRIMARY = """You are the primary use in a wireless communication environment.
Your role is to accept or reject the secondary user proposal for its transmission power based on you sinr value, whether it's under the threshold or not.
Base on your decision you write a critic to inform secondary user to reduce or not its allocation power.

Return JSON matching the schema.
"""

class Secondary(BaseModel):
  allocation: int = Field(..., description="Your allocation power.")

class Primary(BaseModel):
  decision: str = Field(..., description="Whether you ACCEPT or REJECT the secondary user proposal.")
  # issues: List[str] = Field(..., description="The problems with the current allocation.")
  fix_instructions: List[str] = Field(..., description="Order the secondary user on what to do in order to improve your sinr value.")

llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.2)
def secondary_node(state: GraphState) -> GraphState:
  structured_critic = llm.with_structured_output(Secondary)
  resp = structured_critic.invoke([
        SystemMessage(content=PROMPT_SECONDARY),
        HumanMessage(content=f"""
        channel gain
        {state['hss']}

        If critique exists, you may improve the power allocation accordingly.
        Critique:
        {state.get('critique')}
        """)
            ]).allocation
  
  print(f'The secondary user allocation: {resp}')
  state["P2"] = resp
  
  return state


def primary_node(state: GraphState) -> GraphState:
  state['caused_interference'] = calc_interference(state['hsp'], state['P2'])
  gap = allowed_gap(state['caused_interference'], I_max)
  # state['accept'] = "ACCEPT" if gap else "REJECT"

  structured_critic = llm.with_structured_output(Primary)
  resp = structured_critic.invoke([
        SystemMessage(content=PROMPT_PRIMARY),
        HumanMessage(content=f"""REJECT or ACCEPT P2 proposal based on the max interference allowed (The allowed Interference is 1000, adjust the secondary power by reduce/add more power until the allowed gap is True, if not and you see that the interference is under the max you just accept).
        Interference value:
        {state['caused_interference']}
        
        Current gap:
        {gap}
        
        P2 allocatio:
        {state['P2']}
        """)
            ])
  print(f'The primary user decision: {resp.fix_instructions}')
  state["accept"] = resp.decision
  state["critique"] = resp.fix_instructions
  state["iteration"] += 1
  
  return state

def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
  if state["iteration"] == state["max_iter"]:
    return "finalize"
  if state["accept"] == "REJECT":
    return "revise"
  
  return "finalize"

workflow = StateGraph(GraphState)

workflow.add_node("Primary", primary_node)
workflow.add_node("Secondary", secondary_node)

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

hpp, hsp, hps, hss, analytical_best_p2 = gen_channels()
    
initial_state = {
    "P1": P1,
    "P2": 100,      # Will be overwritten by Secondary's first move
    "hpp": hpp,
    "hsp": hsp,
    "hss": hss,
    "sinr": 0.0,
    "caused_interference": 0,
    "iteration": 0,
    "max_iter": 5,
    "accept": "",
    "critique": []
}
print("\n--- Starting LLM Negotiation ---")
result = app.invoke(initial_state)

print("\n--- Final Results ---")
print(f"Final P2 Assigned by LLM: {result['P2']}")
print(f"Analytical Best P2 was:   {analytical_best_p2}")
print(f"Final Primary SINR:       {result['caused_interference']:.2f}")
