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

#load_dotenv()

#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
#os.environ["LANGCHAIN_PROJECT"] = "Test Logging"

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
    data = []
    while len(data) < length:
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

        P = []
        for i in range(5):
            interf_caused = np.sum(H[i]) - H[i][i]

            if interf_caused > 0:
                allowed_p = I_max / interf_caused
            else:
                allowed_p = P_max

            best_P = int(round(min(allowed_p, P_max)))
            P.append(best_P)

        for i in range(5):
            signal = P[i] * H[i][i]

            interference = sum(P[j] * H[j][i] for j in range(5) if j != i)

            sinr = signal / (1.0 + interference)
            se = np.log2(1 + sinr)
            print(f'SE: {se}')
            if 1.0 <= se <= 5.0:
                data.append([H[i][0], H[i][1], H[i][2], H[i][3], H[i][4], P[i]])

    random.shuffle(data)
    return data

class GraphState(TypedDict):
    agent_id: int
    H_matrix: List[List[int]]
    powers: List[int]

    allocation_history: List[List[int]]
    interference_history: List[List[int]]
    delta_hist: List[List[int]]

    iteration: int
    max_iter: int

    individual_critiques: List[dict]
    aggregated_critique: str
    decisions: List[str]

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

def build_train_prompt(train_examples) -> str:
    prompt_s = """You are an individual Transmitter agent in a wireless network.
    Your goal is to choose your optimal transmit power (between 1 and 100) based on your row of channel gains.

    Examples of good power allocations:
    """
    for data_sample in train_examples:
        # Extract row channels and matching best power
        H_matrix, best_p_array = data_sample
        for i in range(5):
            prompt_s += f"If your channel row is {H_matrix[i]}, then your best Power is {best_p_array[i]}\n"

    prompt_s += "\nReturn JSON matching the schema."
    return prompt_s

def allocation(state: GraphState) -> GraphState:
    agent_id = state['agent_id']
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
        state['powers'][agent_id] = new_p
        print(f"  -> Tx {agent_id + 1} Initial Power: {new_p}")

    else:
        prompt = """Adjust your power based on the Aggregated Critique.
        Severity-to-step guide: HIGH=20-30, MEDIUM=10-20, LOW=1-10.
        Provide a single step integer (+ or -) to add or subtract. Do not repeat stalled steps."""

        structured_critic = llm.with_structured_output(ProposersRemainRounds)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"""
            You are Transmitter {agent_id + 1}.
            Your Current Power: {temp_P}
            All Transmitters Current Powers: {state['powers']}
            Aggregated Critique: {state['aggregated_critique']}
            """)
        ])

        step = resp.steps
        new_p = int(max(1, min(100, temp_P + step)))
        state['powers'][agent_id] = new_p
        print(f"  -> Tx {agent_id + 1} Step: {step:+d} | New Power: {new_p}")

    return state

def critique(state: GraphState) -> GraphState:
    agent_id = state['agent_id']
    print(f"[Receiver {agent_id + 1}] Evaluating Interference...")

    H = state['H_matrix']
    P = state['powers']

    # Interference at Rx_agent_id from all Tx_j (j != agent_id)
    interference = sum(P[tx] * H[tx][agent_id] for tx in range(5) if tx != agent_id)
    gap = interference - I_max

    prompt = f"""You are Receiver {agent_id + 1} in a wireless network.
    You evaluate the interference on your channel against the threshold I_max = {I_max}.
    Gap = interference - {I_max}.

    Follow these exact bands based on the Gap:
    1. Gap > 1050: REJECT, DECREASE, HIGH.
    2. 500 <= Gap <= 1050: REJECT, DECREASE, MEDIUM.
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

    # Ensure array padding
    while len(state['individual_critiques']) < 5:
        state['individual_critiques'].append({})
        state['decisions'].append("")

    state['individual_critiques'][agent_id] = {
        "rx_id": agent_id + 1,
        "interference": interference,
        "gap": gap,
        "decision": resp.decision,
        "action": resp.action,
        "severity": resp.severity,
        "critique": resp.critique
    }
    state['decisions'][agent_id] = resp.decision

    print(f"  -> Rx {agent_id + 1} Decision: {resp.decision} (Interference: {interference}, Gap: {gap})")
    return state

def aggregator(state: GraphState) -> GraphState:
    print(f"\n[Aggregator Node] Compiling Reports for Iteration {state['iteration']}...")

    # Record clean snapshots into history
    current_interferences = [c['interference'] for c in state['individual_critiques']]
    state['interference_history'].append(current_interferences)
    state['allocation_history'].append(list(state['powers']))

    prompt = """You are the Critique Aggregator.
    You will read 5 individual feedback critiques from 5 Receivers.
    Your job is to summarize this into ONE actionable paragraph.
    Tell the 5 Transmitters explicitly who needs to INCREASE or DECREASE their power to help the network reach stability."""

    critiques_str = ""
    for i, c in enumerate(state['individual_critiques']):
        critiques_str += f"Rx {i+1}: Decision={c['decision']}, Action={c['action']}, Severity={c['severity']}, Message={c['critique']}\n"

    structured_agg = llm.with_structured_output(AggregatorOutput)
    resp = structured_agg.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=critiques_str)
    ])

    state['aggregated_critique'] = resp.aggregated_critique
    state['iteration'] += 1

    print(f"Aggregated Insight: {resp.aggregated_critique}\n")
    return state

def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
    print("[Finalizer] Checking convergence...")
    if state["iteration"] >= state["max_iter"]:
        print(" -> Maximum iterations reached. Finalizing.")
        return "finalize"

    if "REJECT" in state['decisions']:
        print(" -> Network not converged (REJECT present). Revising allocations.")
        return "revise"

    print(" -> Network converged (ALL ACCEPT). Finalizing.")
    return "finalize"


def make_proposer(agent_id: int):
    def node(state: GraphState) -> GraphState:
        state['agent_id'] = agent_id
        return allocation(state)
    return node

def make_receiver(agent_id: int):
    def node(state: GraphState) -> GraphState:
        state['agent_id'] = agent_id
        return critique(state)
    return node

workflow = StateGraph(GraphState)
for i in range(5):
    workflow.add_node(f"Proposer_{i+1}", make_proposer(i))
    workflow.add_node(f"Receiver_{i+1}", make_receiver(i))

workflow.add_node("Aggregator", aggregator)

# Flow Execution: START -> Proposer_1 -> ... -> Proposer_5 -> Receiver_1 -> ... -> Receiver_5 -> Aggregator
workflow.add_edge(START, "Proposer_1")
workflow.add_edge("Proposer_1", "Proposer_2")
workflow.add_edge("Proposer_2", "Proposer_3")
workflow.add_edge("Proposer_3", "Proposer_4")
workflow.add_edge("Proposer_4", "Proposer_5")

workflow.add_edge("Proposer_5", "Receiver_1")
workflow.add_edge("Receiver_1", "Receiver_2")
workflow.add_edge("Receiver_2", "Receiver_3")
workflow.add_edge("Receiver_3", "Receiver_4")
workflow.add_edge("Receiver_4", "Receiver_5")

workflow.add_edge("Receiver_5", "Aggregator")

# Loop or End condition
workflow.add_conditional_edges(
    "Aggregator",
    finalizer,
    {
        "revise": "Proposer_1",
        "finalize": END,
    }
)

app = workflow.compile()

sample_data = gen_channels(100)
train = sample_data[:80]
test = sample_data[80:]
prmpt_train = build_train_prompt(test)

test_H = test[0][0]

initial_state = {
    "agent_id": 0,
    "H_matrix": test_H,
    "powers": [50, 50, 50, 50, 50],
    "allocation_history": [],
    "interference_history": [],
    "delta_hist": [],
    "iteration": 0,
    "max_iter": 3,
    "individual_critiques": [],
    "aggregated_critique": "",
    "decisions": []
}

output = app.invoke(initial_state)
print("Final Output Powers:", output['powers'])