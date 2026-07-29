""" Last mile problem
    AI identity
"""

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


# load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = "Test Logging"

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

class SingleTargetCritique(BaseModel):
    target_agent: int = Field(description="0-based id of the transmitter this critique is about.")
    decision: Literal["ACCEPT", "REJECT"]
    action: Literal["INCREASE", "DECREASE"]
    severity: Literal["HIGH", "MEDIUM", "LOW", "ACCEPTABLE"]
    critique: str = Field(description="Feedback explicitly restating the step range for the gap.")

class ReceiverCritiqueBatch(BaseModel):
    reasoning: str = Field(description="Your brief reasoning about the gap and each contributor.")
    critiques: List[SingleTargetCritique] = Field(description="Exactly 4 entries, one per other transmitter.")

def critique(state: GraphState, agent_id: int) -> dict:
    if agent_id in state.get('accepted_agents', []):
        return {}

    H = state['H']
    P = state['P']

    interference = sum(P.get(tx, 50) * H[tx][agent_id] for tx in range(5) if tx != agent_id)
    gap = interference - I_max

    contributor_nodes = [tx for tx in range(5) if tx != agent_id]
    contributions = {tx: P.get(tx, 50) * H[tx][agent_id] for tx in contributor_nodes}
    # => {0:1200, 1:980,...} caused interference by each agent

    prompt = f"""You are Receiver {agent_id + 1} in a wireless network.
    You evaluate the interference on your channel against the threshold I_max = {I_max}.

    Follow these exact bands based on the Gap (the distance from the threshold):
    1. Gap > 1000: REJECT, DECREASE, HIGH.
    2. 500 <= Gap <= 1000: REJECT, DECREASE, MEDIUM.
    3. 100 <= Gap <= 499: REJECT, DECREASE, LOW.
    4. Gap <= -500: REJECT, INCREASE, HIGH.
    5. 0 < Gap < 100: ACCEPT.
    6. -499 <= Gap <= 0: ACCEPT.

    Return JSON matching the schema.
    """

    msg = f"Receiver {agent_id + 1}:\nTotal Interference = {interference}\nGap = {gap}\n\n"
    msg += "Contributing transmitters (id: contribution):\n"
    for tx in contributor_nodes:
      msg += f"id {tx}: contributes {contributions[tx]}\n"

    structured_critic = llm.with_structured_output(ReceiverCritiqueBatch)
    resp = structured_critic.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=msg)
    ])

    if resp.decision == "ACCEPT":
        print(f"-> Rx {agent_id + 1} ACCEPT (Interference: {interference:.1f}, Gap: {gap:.1f}) -> locking Tx {agent_id + 1}")
    else:
        print(f"-> Rx {agent_id + 1} Decision: {resp.decision} (Interference: {interference:.1f}, Gap: {gap:.1f})")

    row = {}
    for c in resp.critiques:
        row[c.target_agent] = {
            "decision": c.decision,
            "action": c.action,
            "severity": c.severity,
            "critique": c.critique
        }

    return {"global_critique": {agent_id: row}}

class AggregatorOutput(BaseModel):
    reasoning: str = Field(description="Brief summary of the receivers' critiques for this transmitter.")
    aggregated_critique: str = Field(description="Actionable summary of the received critiques.")
    overall_decision: Literal["ACCEPT", "REJECT"] = Field(
        description="ACCEPT only if the receivers indicate this transmitter needs no further change; REJECT if any meaningful adjustment is still needed."
    )

def aggregator(state: GraphState) -> dict:
    prompt = """You are the Critique Aggregator for ONE transmitter.
    You will read the critiques from each receiver about this transmitter's impact on them.
    Summarize into ONE actionable paragraph, mentioning action and severity explicitly.
    Then decide overall_decision: ACCEPT only if the transmitter genuinely needs no more
    adjustment; REJECT if any receiver still wants a real change.

    Return JSON matching the schema.
    """

    global_critique = state.get('global_critique', {})
    accepted = state.get('accepted_agents', [])

    individual_updates = {}
    newly_accepted = []

    for target in range(5):
        if target in accepted:
            continue

        opinions = []
        for rx in range(5):
            entry = global_critique.get(rx, {}).get(target)
            if entry is not None:
                opinions.append((rx, entry))

        if not opinions:
            continue

        critiques_str = ""
        for rx, entry in opinions:
            critiques_str += f"Rx {rx+1}: Decision={entry['decision']}, Action={entry['action']}, Severity={entry['severity']}, Message={entry['critique']}\n"

        structured_agg = llm.with_structured_output(AggregatorOutput)
        resp = structured_agg.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Transmitter {target + 1}:\n{critiques_str}")
        ])

        individual_updates[target] = resp.aggregated_critique
        print(f"-> Aggregator (Tx {target + 1}): {resp.aggregated_critique}")

        if resp.overall_decision == "ACCEPT":
            newly_accepted.append(target)
            print(f"-> locking Tx {target + 1}")

    return {
        "individual_critique": individual_updates,
        "accepted_agents": newly_accepted,
        "iteration": state.get("iteration", 0) + 1
    }

def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
    print("[Finalizer] Checking convergence...")
    if state.get("iteration", 0) >= 3:
        print("-> Maximum iterations reached. Finalizing.")
        return "finalize"

    accepted_count = len(set(state.get("accepted_agents", [])))

    if accepted_count < 5:
        print(f"-> Network not converged ({accepted_count}/5 transmitters locked). Revising allocations.")
        return "revise"

    print("-> Network converged (ALL 5 transmitters locked). Finalizing.")
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

mae_per_agent = np.zeros(5)
interference_count_per_agent = np.zeros(5)
total_se_pred = np.zeros(5)
total_se_true = np.zeros(5)
final_interferences = []

MAE = {}
caused_interference = 0
se_pred = {}
se_true = {}
all_interference = []

for i in range(len(test_h)):
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
  current_sample_interferences = []
  current_interferences = []

  for r in range(5):
    interference = sum(output["P"][j] * test_h[i][j][r] for j in range(5) if j != r)
    current_interferences.append(interference)

    mae_per_agent[r] += abs(output["P"][r] - test_p[i][r])
    if interference > I_max:
      interference_count_per_agent[r] += 1

    sinr_pred = interference
    # True interference from test_p
    sinr_true = sum(test_p[i][j] * test_h[i][j][r] for j in range(5) if j != r)

    se_p = math.log2(1 + (output["P"][r] * test_h[i][r][r]) / (1 + sinr_pred))
    se_t = math.log2(1 + (test_p[i][r] * test_h[i][r][r]) / (1 + sinr_true))

    total_se_pred[r] += se_p
    total_se_true[r] += se_t
    if i == len(test_h) - 1:
        current_sample_interferences.append(output["P"][r])

    if i == len(test_h) - 1:
        final_interferences = current_sample_interferences

    
mae_per_agent /= len(test_h)
interference_rate = interference_count_per_agent / len(test_h)
avg_se_pred = total_se_pred / len(test_h)
avg_se_true = total_se_true / len(test_h)

print(f"Average MAE per Agent: {mae_per_agent}")
print(f"Interference Rate per Agent: {interference_rate}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

agents = [f'Agent {i+1}' for i in range(5)]
x = np.arange(len(agents))

# Plot 1: SE Comparison (Averages over test set)
ax1.bar(x - 0.2, avg_se_pred, 0.4, label='Avg Predicted SE', color='royalblue')
ax1.bar(x + 0.2, avg_se_true, 0.4, label='Avg True SE', color='orange')
ax1.set_xticks(x)
ax1.set_xticklabels(agents)
ax1.set_ylabel('Spectral Efficiency (bits/s/Hz)')
ax1.set_title('Average SE: Predicted vs True')
ax1.legend()

# Plot 2: Interference vs Threshold (using last sample as representative)
ax2.bar(agents, final_interferences, color='lightgreen', label='Final Sample Interference')
ax2.axhline(y=I_max, color='red', linestyle='--', linewidth=2, label=f'Threshold (I_max={I_max})')
ax2.set_ylabel('Interference Level')
ax2.set_title('Interference Levels vs. Threshold')
ax2.legend()

plt.tight_layout()
plt.show()