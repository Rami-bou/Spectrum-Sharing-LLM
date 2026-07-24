import random
import numpy as np
import math
from typing import List, Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.channels import Noop

# --- Environment Setup ---
f = 2e9
c = 3e8
wave = c / f
P_max = 100.0
I_max = 1000
scale_factor = 1e8

random.seed(10)

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


# --- Graph State ---
class GraphState(TypedDict):
    # Removed agent_id from shared state as it's passed directly to nodes
    H_matrix: Annotated[List[List[int]], Noop()] # Mark H_matrix as read-only for parallel steps
    powers: List[int]

    allocation_history: List[List[int]]
    interference_history: List[List[int]]
    # Removed delta_hist as it is not used
    steps_history: List[List[int]]

    iteration: int
    max_iter: int

    individual_critiques: List[dict]
    aggregated_critique: str
    decisions: List[str]

    accepted_agents: List[bool]   # Tracks which agents are accepted/disabled
    final_allocation: List[int]  # Stores locked powers for accepted agents


llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)


# --- Pydantic Schemas ---
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


# --- Prompt Builder ---
def build_train_prompt(train_examples) -> str:
    prompt_s = """You are an individual Transmitter agent in a wireless network.
    Your goal is to choose your optimal transmit power (between 1 and 100) based on your row of channel gains.

    Examples of channel matrices:
    """
    for data_sample in train_examples[:5]:
        prompt_s += f"Sample Matrix: {data_sample}\n"

    prompt_s += "\nReturn JSON matching the schema."
    return prompt_s


# --- Nodes ---
def allocation(state: GraphState, agent_id: int) -> GraphState:
    # agent_id is now passed as an argument, no longer fetched from state

    # DISABLE/FREEZE: If agent is already accepted, lock its power and skip LLM
    if state['accepted_agents'][agent_id]:
        print(f"[Proposer {agent_id + 1}] ALREADY ACCEPTED. Power locked at {state['powers'][agent_id]}. Skipping proposal.")
        return state

    print(f"[Proposer {agent_id + 1}] Working on Iteration {state['iteration']}...")

    H = state['H_matrix']
    P = state['powers']

    temp_H = H[agent_id]
    temp_P = P[agent_id]

    if not state['aggregated_critique']: # Only make an initial proposal if there's no critique yet
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
            Your Step History: {state['steps_history'][agent_id]}
            """)
        ])

        step = resp.steps
        state['steps_history'][agent_id].append(step)
        new_p = int(max(1, min(100, temp_P + step)))
        state['powers'][agent_id] = new_p
        print(f"  -> Tx {agent_id + 1} Step: {step:+d} | New Power: {new_p}")

    return state


def critique(state: GraphState, agent_id: int) -> GraphState:
    # agent_id is now passed as an argument, no longer fetched from state

    H = state['H_matrix']
    P = state['powers']

    interference = sum(P[tx] * H[tx][agent_id] for tx in range(5) if tx != agent_id)
    gap = interference - I_max

    # DISABLE/FREEZE: If agent was already accepted, keep ACCEPT status
    if state['accepted_agents'][agent_id]:
        print(f"[Receiver {agent_id + 1}] ALREADY ACCEPTED. Maintaining ACCEPT (Interference: {interference}, Gap: {gap}).")
        state['decisions'][agent_id] = "ACCEPT"
        state['individual_critiques'][agent_id] = {
            "rx_id": agent_id + 1,
            "interference": interference,
            "gap": gap,
            "decision": "ACCEPT",
            "action": "ACCEPTABLE",
            "severity": "ACCEPTABLE",
            "critique": f"Receiver {agent_id + 1} is locked and accepted."
        }
        return state

    print(f"[Receiver {agent_id + 1}] Evaluating Interference...")

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

    # If accepted, lock agent and store final power
    if resp.decision == "ACCEPT":
        state['accepted_agents'][agent_id] = True
        state['final_allocation'][agent_id] = state['powers'][agent_id]
        print(f"  *** [Receiver {agent_id + 1}] PROPOSAL ACCEPTED! Locking power at {state['powers'][agent_id]} ***")

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

    current_interferences = [c.get('interference', 0) for c in state['individual_critiques']]
    state['interference_history'].append(current_interferences)
    state['allocation_history'].append(list(state['powers']))

    prompt = """You are the Critique Aggregator.
    You will read 5 individual feedback critiques from 5 Receivers.
    Your job is to summarize this into ONE actionable paragraph.
    Tell the Transmitters explicitly who needs to INCREASE or DECREASE their power to help the network reach stability.
    Ignore agents that are already ACCEPTED/locked."""

    critiques_str = ""
    for i, c in enumerate(state['individual_critiques']):
        status = "LOCKED" if state['accepted_agents'][i] else "ACTIVE"
        critiques_str += f"Rx {i+1} [{status}]: Decision={c.get('decision')}, Action={c.get('action')}, Severity={c.get('severity')}, Message={c.get('critique')}\n"

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

    if all(state['accepted_agents']):
        print(" -> All 5 agents ACCEPTED. Network fully converged. Finalizing.")
        return "finalize"

    if "REJECT" in state['decisions']:
        print(" -> Network not fully converged (REJECT present). Revising active allocations.")
        return "revise"

    print(" -> Network converged. Finalizing.")
    return "finalize"


# --- Node Creator Wrappers ---
def make_proposer(agent_id: int):
    def node(state: GraphState) -> GraphState:
        # Pass agent_id directly to allocation, removed state['agent_id'] assignment
        return allocation(state, agent_id)
    return node

def make_receiver(agent_id: int):
    def node(state: GraphState) -> GraphState:
        # Pass agent_id directly to critique, removed state['agent_id'] assignment
        return critique(state, agent_id)
    return node

def start_node(state: GraphState) -> GraphState:
    print(f"\n==================== [Iteration {state['iteration']}] ====================")
    return state

def middle_node(state: GraphState) -> GraphState:
    return state


# --- Graph Construction ---
workflow = StateGraph(GraphState)

workflow.add_node("Start", start_node)
workflow.add_node("Middle", middle_node)

for i in range(5):
    workflow.add_node(f"Proposer_{i+1}", make_proposer(i))
    workflow.add_node(f"Receiver_{i+1}", make_receiver(i))

workflow.add_node("Aggregator", aggregator)

# Fan-out: Start -> 5 Parallel Proposers -> Middle
workflow.add_edge(START, "Start")
for i in range(1, 6):
    workflow.add_edge("Start", f"Proposer_{i}")
    workflow.add_edge(f"Proposer_{i}", "Middle")

# Fan-out: Middle -> 5 Parallel Receivers -> Aggregator
for i in range(1, 6):
    workflow.add_edge("Middle", f"Receiver_{i}")
    workflow.add_edge(f"Receiver_{i}", "Aggregator")

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
test = sample_data[80:]

prmpt_train = build_train_prompt(train)


test_H_matrix = test[0]

initial_state = {
    # Removed 'agent_id': 0 as it's no longer part of the shared state
    "H_matrix": test_H_matrix,
    "powers": [50, 50, 50, 50, 50],
    "allocation_history": [],
    "interference_history": [],
    "steps_history": [[], [], [], [], []],
    "iteration": 0,
    "max_iter": 4,
    "individual_critiques": [{}, {}, {}, {}, {}],
    "aggregated_critique": "",
    "decisions": ["", "", "", "", ""],
    "accepted_agents": [False, False, False, False, False],
    "final_allocation": [0, 0, 0, 0, 0]
}

output = app.invoke(initial_state)

print("\n==================== RESULTS ====================")
print("Final Powers Array:     ", output['powers'])
print("Final Allocation Locked:", output['final_allocation'])
print("Accepted Agents Mask:   ", output['accepted_agents'])