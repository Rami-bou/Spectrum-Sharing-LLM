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
# P1 = 100 # Not used in the function, can be removed or commented out
scale_factor = 1e8
# possible_P2 = [] # Not used in the function, can be removed or commented out

random.seed(10)
# number of nodes
N = 3
# Global H and P are removed to make the gen_channels function self-contained
# H = []
# P = []

def gen_channels(length):
    generated_H_matrices = []
    generated_target_powers = []

    while len(generated_H_matrices) < length:
        current_H_matrix = []
        for i in range(5):
            row = []
            for j in range(5):
                tx = [random.uniform(0, 30), random.uniform(0, 50)]
                rx = [random.uniform(31, 60), random.uniform(0, 50)]
                d = np.sqrt((tx[0] - rx[0])**2 + (tx[1] - rx[1])**2)
                h = (wave / (4 * np.pi * d))**2
                h_normal = int(round(h * scale_factor))
                row.append(h_normal)
            current_H_matrix.append(row)

        for i in range(5):
            current_H_matrix[i][i] += 100

        current_target_powers = []
        valid_sample = True

        for i in range(5):
            interf_gain = sum(current_H_matrix[i][j] for j in range(5) if j != i)
            allowed_p = I_max / interf_gain if interf_gain > 0 else P_max
            best_P = int(round(min(allowed_p, P_max)))
            best_P = max(1, best_P)


            interf_received = sum(best_P * current_H_matrix[j][i] for j in range(5) if j != i)

            se = np.log2(1 + (best_P * current_H_matrix[i][i] / (1 + interf_received)))

            if not (0.5 <= se <= 8.0):
                valid_sample = False
                break

            current_target_powers.append(best_P)

        if valid_sample:
            generated_H_matrices.append(current_H_matrix)
            generated_target_powers.append(current_target_powers)

    return generated_H_matrices, generated_target_powers


result_h, result_p = gen_channels(2)
for i in range(len(result_h)):
    print(f"H{i+1}:")
    for row in result_h[i]:
        print(row)
    print(f"P{i+1}: {result_p[i]}")

def merge_dict(existing: dict, update: dict) -> dict:
    merged = existing.copy() if existing else {}
    if update:
        merged.update(update)
    return merged

class GraphState(TypedDict):
    H: List[List[int]]
    P: Annotated[dict, merge_dict]  # Changed to dict for safe parallel updates
    global_critique: Annotated[dict, merge_dict]
    individual_critique: Annotated[dict, merge_dict]
    accepted_agents: Annotated[list, operator.add]
    steps_history: Annotated[dict, merge_dict]
    iteration: int

llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)

class ProposersFirstRound(BaseModel):
    reasoning: str = Field(description="Your brief reasoning for initial power selection.")
    powers: int = Field(description="Your single power allocation value between 1 and 100.")

class ProposersRemainRounds(BaseModel):
    reasoning: str = Field(description="Your brief reasoning for adjusting power.")
    steps: int = Field(description="Single delta integer (+ or -) to adjust your power.")

def allocation(state: GraphState, agent_id) -> dict: # Must return dict
    if agent_id in state.get('accepted_agents', []):
        return {} # Must return empty dict, not None

    agent_h_row = state['H'][agent_id]
    agent_p_val = state['P'].get(agent_id, 50)

    if not state.get('individual_critique'):
        prompt_s = """You are an individual Transmitter agent in a wireless network.
        Your goal is to choose your optimal transmit power (between 1 and 100) based on your row of channel gains.

        Examples of good power allocations:
        """
        # FIX: Loop through the rows safely without crashing on integer casting
        for i in range(len(state['H'])):
            prompt_s += f"If your channel row is {state['H'][i]}, then your best Power is {state['P'].get(i, 50)}\n"

        prompt_s += "\nReturn JSON matching the schema."

        structured_critic = llm.with_structured_output(ProposersFirstRound)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt_s),
            HumanMessage(content=f"""
            You are Transmitter {agent_id + 1}.
            Your Channel Gains (Row {agent_id + 1}): {agent_h_row}
            Propose your initial Power allocation (1 to 100):
            """)
        ])

        new_p = int(max(1, min(100, resp.powers)))
        print(f"-> Tx {agent_id + 1} Initial Power: {new_p}")

        # Return ONLY the dictionary delta (no direct state mutation)
        return {"P": {agent_id: new_p}}

    else:
        prompt = """Adjust your power based on the Aggregated Critique.
        Severity-to-step guide: HIGH=20-30, MEDIUM=10-20, LOW=1-10.
        Provide a single step integer (if action is INCREASE you provide a positive step, otherwise you provide a negative step) to add or subtract. Do not repeat stalled steps, for this check steps history.

        Return JSON matching the schema.
        """

        own_steps = state.get('steps_history', {}).get(agent_id, [])
        structured_critic = llm.with_structured_output(ProposersRemainRounds)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"""
            You are Transmitter {agent_id + 1}.
            Your Current Power: {agent_p_val}
            Aggregated Critique: {state['individual_critique'].get(agent_id, "")}
            Your Own Steps History: {own_steps}
            """)
        ])

        step = resp.steps
        new_p = int(max(1, min(100, agent_p_val + step)))
        print(f"-> Tx {agent_id + 1} Step: {step:+d} | New Power: {new_p}")

        # Return ONLY the dictionary delta
        return {
            "P": {agent_id: new_p},
            "steps_history": {agent_id: own_steps + [step]}
        }

class SingleReceiverCritique(BaseModel):
    reasoning: str = Field(description="Your brief reasoning for the decision.")
    decision: Literal["ACCEPT", "REJECT"]
    action: Literal["INCREASE", "DECREASE"]
    severity: Literal["HIGH", "MEDIUM", "LOW", "ACCEPTABLE"]
    critique: str = Field(description="Feedback explicitly restating the step range for the gap.")

def critique(state: GraphState, agent_id: int) -> dict:
    if agent_id in state.get('accepted_agents', []):
        return {}

    H = state['H']
    P = state['P']

    interference = sum(P.get(tx, 50) * H[tx][agent_id] for tx in range(5) if tx != agent_id)
    gap = interference - I_max

    prompt = f"""You are Receiver {agent_id + 1} in a wireless network.
    You evaluate the interference on your channel against the threshold I_max = {I_max}.
    Gap = interference - {I_max}.

    Follow these exact bands based on the Gap:
    1. Gap > 1000: REJECT, DECREASE, HIGH.
    2. 500 <= Gap <= 1000: REJECT, DECREASE, MEDIUM.
    3. 100 <= Gap <= 499: REJECT, DECREASE, LOW.
    4. Gap <= -500: REJECT, INCREASE, HIGH.
    5. 0 < Gap < 100: ACCEPT.
    6. -499 <= Gap <= 0: ACCEPT.

    Return JSON matching the schema.
    """

    msg = f"Receiver {agent_id + 1}:\nReceived Interference = {interference}\nGap = {gap}\n"

    structured_critic = llm.with_structured_output(SingleReceiverCritique)
    resp = structured_critic.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=msg)
    ])

    if resp.decision == "ACCEPT":
        print(f"-> Rx {agent_id + 1} ACCEPT (Interference: {interference:.1f}, Gap: {gap:.1f}) -> locking Tx {agent_id + 1}")
    else:
        print(f"-> Rx {agent_id + 1} Decision: {resp.decision} (Interference: {interference:.1f}, Gap: {gap:.1f})")

    update_dict = {
        "global_critique": {
            agent_id: {
                "decision": resp.decision,
                "action": resp.action,
                "severity": resp.severity,
                "critique": resp.critique
            }
        }
    }

    if resp.decision == "ACCEPT":
        update_dict["accepted_agents"] = [agent_id]

    return update_dict

class AggregatorOutput(BaseModel):
    reasoning: str = Field(description="Brief summary of overall network state.")
    aggregated_critique: str = Field(description="Actionable summary of the 5 received critiques.")

def aggregator(state: GraphState) -> dict: # Must return dict, NOT GraphState
    prompt = """You are the Critique Aggregator.
    You will read 5 individual feedback critiques.
    Your job is to summarize this into ONE actionable paragraph.
    You mention explicitly the decision, action to do and its severity.

    Return JSON matching the schema.
    """

    critiques_str = ""
    for i in range(5):
        c = state.get('global_critique', {}).get(i, {})
        decision = c.get('decision', 'N/A')
        action = c.get('action', 'N/A')
        severity = c.get('severity', 'N/A')
        message = c.get('critique', 'No critique provided')
        critiques_str += f"Rx {i+1}: Decision={decision}, Action={action}, Severity={severity}, Message={message}\n"

    structured_agg = llm.with_structured_output(AggregatorOutput)
    resp = structured_agg.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=critiques_str)
    ])

    updates = {}
    for i in range(5):
        updates[i] = resp.aggregated_critique

    # Return ONLY the delta
    return {
        "individual_critique": updates,
        "iteration": state.get("iteration", 0) + 1
    }

def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
    print("[Finalizer] Checking convergence...")
    if state.get("iteration", 0) >= 3:
        print(" -> Maximum iterations reached. Finalizing.")
        return "finalize"

    # Check the latest decisions dynamically
    current_decisions = [
        c.get("decision") for c in state.get("global_critique", {}).values()
    ]

    if "REJECT" in current_decisions:
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

def build_train_prompt(h, p, id) -> str:
    prompt_s = """You are an individual Transmitter agent in a wireless network.
    Your goal is to choose your optimal transmit power (between 1 and 100) based on your row of channel gains.

    Examples of good power allocations:
    """
    for i in range(len(h)):
      prompt_s += f"If your channel row is {list(h[id][i])}, then your best Power is {p[id][i]}\n"

    prompt_s += "\nReturn JSON matching the schema."
    return prompt_s

import matplotlib.pyplot as plt
import math

result_h, result_p = gen_channels(100)
train_h = result_h[:80]
train_p = result_p[:80]
test_h = result_h[80:]
test_p = result_p[80:]
states = []

MAE = {}
caused_interference = 0
se_pred = {}
se_true = {}
all_interference = []

# Run for 1 test case
for i in range(1):
  MAE[i] = 0
  se_pred[i] = 0
  se_true[i] = 0
  initial_state = {
          "H": test_h[i],
          "P": {j: test_p[i][j] for j in range(5)},
          "global_critique": {},
          "individual_critique": {},
          "accepted_agents": [],
          "steps_history": {},
          "iteration": 0
      }
  states.append(initial_state)
  output = app.invoke(initial_state)

  current_interferences = []
  for r in range(5):
    # Calculate interference for receiver r
    interference = sum(output["P"][j] * test_h[i][j][r] for j in range(5) if j != r)
    current_interferences.append(interference)

    MAE[i] += abs(output["P"][r] - test_p[i][r])
    if interference > I_max:
      caused_interference += 1

    sinr_pred = interference
    # True interference from test_p
    sinr_true = sum(test_p[i][j] * test_h[i][j][r] for j in range(5) if j != r)

    se_pred[i] += math.log2(1 + (output["P"][r] * test_h[i][r][r]) / (1 + sinr_pred))
    se_true[i] += math.log2(1 + (test_p[i][r] * test_h[i][r][r]) / (1 + sinr_true))

  all_interference.append(current_interferences)

  print(f'P: {list(output["P"].values())}')
  print(f'True P: {test_p[i]}')
  print(f'MAE: {MAE[i] / 5}')

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: SE Comparison
agents = [f'Agent {i+1}' for i in range(5)]
# We calculate per-agent SE for the last run for plotting purposes
last_se_pred = []
last_se_true = []
i = 0 # Using first test case results
for r in range(5):
    interf_p = sum(output["P"][j] * test_h[i][j][r] for j in range(5) if j != r)
    interf_t = sum(test_p[i][j] * test_h[i][j][r] for j in range(5) if j != r)
    last_se_pred.append(math.log2(1 + (output["P"][r] * test_h[i][r][r]) / (1 + interf_p)))
    last_se_true.append(math.log2(1 + (test_p[i][r] * test_h[i][r][r]) / (1 + interf_t)))

ax1.bar(np.arange(5) - 0.2, last_se_pred, 0.4, label='Predicted SE')
ax1.bar(np.arange(5) + 0.2, last_se_true, 0.4, label='True SE')
ax1.set_xticks(range(5))
ax1.set_xticklabels(agents)
ax1.set_ylabel('Spectral Efficiency')
ax1.set_title('SE Prediction vs True')
ax1.legend()

# Plot 2: Interference vs Threshold
ax2.bar(agents, all_interference[0], color='skyblue', label='Interference')
ax2.axhline(y=I_max, color='r', linestyle='--', label='Threshold (I_max)')
ax2.set_ylabel('Interference Level')
ax2.set_title('Interference per Receiver')
ax2.legend()

plt.tight_layout()
plt.show()