# -*- coding: utf-8 -*-
"""5Agents_spec_alloc.ipynb"""

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
from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
import math
from dotenv import load_dotenv

f = 2e9
c = 3e8
wave = c / f
P_max = 100.0

I_max = 1000
P1 = 100
scale_factor = 1e8
possible_P2 = []

random.seed(10)

H = []
P = []

def gen_channels(length):
    dataset = []
    while len(dataset) < length:
        H = []
        for i in range(5):
            row = []
            for j in range(5):
                tx = [random.uniform(0, 30), random.uniform(0, 50)]
                rx = [random.uniform(31, 60), random.uniform(0, 50)]
                d = np.sqrt((tx[0]-rx[0])**2 + (tx[1]-rx[1])**2)
                h = (wave / (4 * np.pi * d))**2
                h_normal = int(round(h * scale_factor))
                row.append(h_normal)
            H.append(row)

        H = np.array(H)
        for i in range(5):
            H[i][i] += 100

        dataset.append(H.tolist())

    random.shuffle(dataset)
    return dataset


# --- Reducers: let all 5 Proposer/Receiver branches write into the same
# shared dict in one super-step without conflicting. Each branch contributes
# only its own agent_id key; the reducer merges them together. ---
def merge_dict(current: dict, update: dict) -> dict:
    merged = dict(current) if current else {}
    merged.update(update)
    return merged

def merge_steps(current: dict, update: dict) -> dict:
    merged = {k: list(v) for k, v in (current or {}).items()}
    for agent_id, step in update.items():
        merged.setdefault(agent_id, []).append(step)
    return merged


class GraphState(TypedDict):
    H_matrix: List[List[int]]

    # Keyed by agent_id (0-4). Annotated + reducer = safe for all 5
    # Proposer/Receiver branches to write in the same super-step.
    powers: Annotated[Dict[int, int], merge_dict]
    individual_critiques: Annotated[Dict[int, dict], merge_dict]
    decisions: Annotated[Dict[int, str], merge_dict]
    final_allocation: Annotated[Dict[int, int], merge_dict]
    steps_history: Annotated[Dict[int, List[int]], merge_steps]

    allocation_history: List[List[int]]
    interference_history: List[List[int]]
    delta_hist: List[List[int]]

    iteration: int
    max_iter: int

    aggregated_critique: str


llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)

class ProposersFirstRound(BaseModel):
    reasoning: str = Field(description="Your brief reasoning for initial power selection.")
    powers: int = Field(description="Your single power allocation value between 1 and 100.")

class ProposersRemainRounds(BaseModel):
    reasoning: str = Field(description="Your brief reasoning for adjusting power.")
    steps: int = Field(description="Single delta integer (+ or -) to adjust your power.")

class SingleReceiverCritique(BaseModel):
    reasoning: str = Field(description="Your brief reasoning for the decision.")
    decision: Literal["ACCEPT", "REJECT"]
    action: Literal["INCREASE", "DECREASE"]
    severity: Literal["HIGH", "MEDIUM", "LOW", "ACCEPTABLE"]
    critique: str = Field(description="Feedback explicitly restating the step range for the gap.")

class AggregatorOutput(BaseModel):
    reasoning: str = Field(description="Brief summary of overall network state.")
    aggregated_critique: str = Field(description="Actionable summary telling Tx1-Tx5 who should increase or decrease power.")


def build_train_prompt(train_matrices, num_examples: int = 10) -> str:
    """
    gen_channels() returns raw 5x5 matrices only (no precomputed 'best power'
    label anymore). We derive a quick heuristic suggestion per row here just
    for few-shot prompting: lower power when a transmitter's total outgoing
    gain to the other 4 receivers is high. Capped to num_examples rows so the
    prompt doesn't balloon on a local 14B model's context.
    """
    prompt_s = """You are an individual Transmitter agent in a wireless network.
    Your goal is to choose your optimal transmit power (between 1 and 100) based on your row of channel gains.

    Examples of good power allocations:
    """
    for H_matrix in train_matrices[:num_examples]:
        for i in range(5):
            row = H_matrix[i]
            self_gain = row[i]
            interf_caused = sum(row) - self_gain
            suggested_p = P_max if interf_caused <= 0 else min(P_max, I_max / interf_caused)
            prompt_s += f"If your channel row is {row}, then a reasonable Power is {int(round(suggested_p))}\n"

    prompt_s += "\nReturn JSON matching the schema."
    return prompt_s


def allocation(state: GraphState, agent_id: int) -> dict:
    if agent_id in state['final_allocation']:
        print(f"[Proposer {agent_id + 1}] Already locked at {state['final_allocation'][agent_id]}, skipping.")
        return {}

    print(f"[Proposer {agent_id + 1}] Working on Iteration {state['iteration']}...")

    H = state['H_matrix']
    P = state['powers']

    temp_H = H[agent_id]
    temp_P = P[agent_id]

    if not state['aggregated_critique']:
        prompt = prmpt_train
        structured_critic = llm.with_structured_output(ProposersFirstRound)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"""
            You are Transmitter {agent_id + 1}.
            Your Channel Gains (Row {agent_id + 1}): {temp_H}
            Propose your initial Power allocation (1 to 100):
            """)
        ])

        new_p = int(max(1, min(100, resp.powers)))
        print(f"  -> Tx {agent_id + 1} Initial Power: {new_p}")
        return {"powers": {agent_id: new_p}}

    else:
        prompt = """Adjust your power based on the Aggregated Critique.
        Severity-to-step guide: HIGH=20-30, MEDIUM=10-20, LOW=1-10.
        Provide a single step integer (+ or -) to add or subtract. Do not repeat stalled steps, for this check steps history."""

        own_steps = state['steps_history'].get(agent_id, [])
        structured_critic = llm.with_structured_output(ProposersRemainRounds)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"""
            You are Transmitter {agent_id + 1}.
            Your Current Power: {temp_P}
            All Transmitters Current Powers: {[P[i] for i in range(5)]}
            Aggregated Critique: {state['aggregated_critique']}
            Your Own Steps History: {own_steps}
            """)
        ])

        step = resp.steps
        new_p = int(max(1, min(100, temp_P + step)))
        print(f"  -> Tx {agent_id + 1} Step: {step:+d} | New Power: {new_p}")
        return {"powers": {agent_id: new_p}, "steps_history": {agent_id: step}}


def critique(state: GraphState, agent_id: int) -> dict:
    if agent_id in state['final_allocation']:
        return {}

    print(f"[Receiver {agent_id + 1}] Evaluating Interference...")

    H = state['H_matrix']
    P = state['powers']

    interference = sum(P[tx] * H[tx][agent_id] for tx in range(5) if tx != agent_id)
    gap = interference - I_max

    prompt = f"""You are Receiver {agent_id + 1} in a wireless network.
    You evaluate the interference on your channel against the threshold I_max = {I_max}.
    Gap = interference - {I_max}.

    Follow these exact bands based on the Gap:
    1. Gap > 1000: REJECT, DECREASE, HIGH.
    2. 500 <= Gap <= 1000: REJECT, DECREASE, MEDIUM.
    3. 10 <= Gap <= 499: REJECT, DECREASE, LOW.
    4. Gap <= -500: REJECT, INCREASE, HIGH.
    5. 0 < Gap <= 9: ACCEPT.
    6. -499 <= Gap <= 0: ACCEPT.

    Return JSON matching the schema."""

    msg = f"Receiver {agent_id + 1}:\nReceived Interference = {interference}\nGap = {gap}\n"

    structured_critic = llm.with_structured_output(SingleReceiverCritique)
    resp = structured_critic.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=msg)
    ])

    update = {
        "individual_critiques": {
            agent_id: {
                "rx_id": agent_id + 1,
                "interference": interference,
                "gap": gap,
                "decision": resp.decision,
                "action": resp.action,
                "severity": resp.severity,
                "critique": resp.critique,
            }
        },
        "decisions": {agent_id: resp.decision},
    }

    if resp.decision == "ACCEPT":
        update["final_allocation"] = {agent_id: P[agent_id]}
        print(f"  -> Rx {agent_id + 1} ACCEPT (Interference: {interference}, Gap: {gap}) -> locking Tx {agent_id + 1} at {P[agent_id]}")
    else:
        print(f"  -> Rx {agent_id + 1} Decision: {resp.decision} (Interference: {interference}, Gap: {gap})")

    return update


def aggregator(state: GraphState) -> dict:
    print(f"\n[Aggregator Node] Compiling Reports for Iteration {state['iteration']}...")

    current_interferences = [state['individual_critiques'][i]['interference'] for i in range(5)]
    new_interference_history = state['interference_history'] + [current_interferences]
    new_allocation_history = state['allocation_history'] + [[state['powers'][i] for i in range(5)]]

    prompt = """You are the Critique Aggregator.
    You will read 5 individual feedback critiques from 5 Receivers.
    Your job is to summarize this into ONE actionable paragraph.
    Tell the 5 Transmitters explicitly who needs to INCREASE or DECREASE their power to help the network reach stability."""

    critiques_str = ""
    for i in range(5):
        c = state['individual_critiques'][i]
        critiques_str += f"Rx {i+1}: Decision={c['decision']}, Action={c['action']}, Severity={c['severity']}, Message={c['critique']}\n"

    structured_agg = llm.with_structured_output(AggregatorOutput)
    resp = structured_agg.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=critiques_str)
    ])

    print(f"Aggregated Insight: {resp.aggregated_critique}\n")

    return {
        "aggregated_critique": resp.aggregated_critique,
        "iteration": state['iteration'] + 1,
        "interference_history": new_interference_history,
        "allocation_history": new_allocation_history,
    }


def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
    print("[Finalizer] Checking convergence...")
    if len(state['final_allocation']) >= 5:
        print(" -> All 5 Transmitters locked in. Finalizing.")
        return "finalize"

    if state["iteration"] >= state["max_iter"]:
        print(" -> Maximum iterations reached. Finalizing.")
        return "finalize"

    print(f" -> {5 - len(state['final_allocation'])} Transmitter(s) still negotiating. Revising.")
    return "revise"


def make_proposer(agent_id: int):
    def node(state: GraphState) -> dict:
        return allocation(state, agent_id)
    return node

def make_receiver(agent_id: int):
    def node(state: GraphState) -> dict:
        return critique(state, agent_id)
    return node

def start_node(state: GraphState) -> dict:
    print(f"\n[Start Node] Dispatching Iteration {state['iteration']}...")
    return {}

def middle_node(state: GraphState) -> dict:
    return {}


workflow = StateGraph(GraphState)

workflow.add_node("Start", start_node)
workflow.add_node("Middle", middle_node)

for i in range(5):
    workflow.add_node(f"Proposer_{i+1}", make_proposer(i))
    workflow.add_node(f"Receiver_{i+1}", make_receiver(i))

workflow.add_node("Aggregator", aggregator)

workflow.add_edge(START, "Start")
for i in range(5):
    workflow.add_edge("Start", f"Proposer_{i+1}")
    workflow.add_edge(f"Proposer_{i+1}", "Middle")

for i in range(5):
    workflow.add_edge("Middle", f"Receiver_{i+1}")
    workflow.add_edge(f"Receiver_{i+1}", "Aggregator")

workflow.add_conditional_edges(
    "Aggregator",
    finalizer,
    {
        "revise": "Start",
        "finalize": END,
    }
)

app = workflow.compile()

try:
    from IPython.display import Image, display
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception as e:
    print("Graph visualization skipped:", e)


sample_data = gen_channels(100)
train = sample_data[:80]
test_H_matrix = sample_data[80]

prmpt_train = build_train_prompt(train)

initial_state = {
    "H_matrix": test_H_matrix,
    "powers": {i: 50 for i in range(5)},
    "allocation_history": [],
    "interference_history": [],
    "delta_hist": [],
    "iteration": 0,
    "max_iter": 3,
    "individual_critiques": {},
    "aggregated_critique": "",
    "decisions": {},
    "final_allocation": {},
    "steps_history": {},
}

output = app.invoke(initial_state)
print("Final Powers:", {i: output['powers'][i] for i in range(5)})
print("True Powers:", {test_H_matrix[i][4] for i in range(5)})