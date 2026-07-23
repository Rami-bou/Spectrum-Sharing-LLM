import matplotlib.pyplot as plt
import random
import numpy as np
import math
import itertools
from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# --- Environment Setup ---
f = 2e9
c = 3e8
wave = c / f
P_max = 100
I_max = 1000
scale_factor = 1e8

random.seed(10)

llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)

# --- Channel & Data Generation for 5 Agents ---
data = []

def gen_channels_5(length):
    """
    Generates a 5x5 channel matrix H for 5 Tx-Rx pairs.
    H[i][j] is the channel gain from Tx_i to Rx_j.
    """
    global data
    while len(data) < length:
        H = []
        for i in range(5):
            row = []
            for j in range(5):
                # Direct channels (i==j) are closer physically
                d = random.uniform(2, 6) if i == j else random.uniform(10, 30)
                h_val = (wave / (4 * math.pi * d))**2
                h_norm = int(round(h_val * scale_factor))
                row.append(h_norm)
            H.append(row)
            
        # Find analytical best power using grid search for baseline/training
        best_p = [10, 10, 10, 10, 10]
        max_sum_se = -1
        
        # Coarse grid search to save generation time: P = {20, 40, 60, 80, 100}
        for p in itertools.product([20, 40, 60, 80, 100], repeat=5):
            valid = True
            for rx in range(5):
                interf = sum(p[tx] * H[tx][rx] for tx in range(5) if tx != rx)
                if interf > I_max:
                    valid = False
                    break
            if not valid:
                continue
                
            sum_se = 0
            for rx in range(5):
                signal = p[rx] * H[rx][rx]
                interf = sum(p[tx] * H[tx][rx] for tx in range(5) if tx != rx)
                sum_se += np.log2(1 + signal / (1 + interf))
                
            if sum_se > max_sum_se:
                max_sum_se = sum_se
                best_p = list(p)
                
        # Only keep if a valid configuration below interference limit was found
        if max_sum_se != -1:
            data.append((H, best_p))

    return data


# --- Graph State & Pydantic Models ---
class GraphState(TypedDict):
    H_matrix: List[List[int]]
    powers: List[int]                     # [P1, P2, P3, P4, P5]
    
    allocation_history: List[List[int]]
    interference_history: List[List[int]] # [ [I1..I5], ... ]
    delta_hist: List[List[int]]
    
    iteration: int
    max_iter: int

    individual_critiques: List[dict]
    aggregated_critique: str
    decisions: List[str]


class SingleReceiverCritique(BaseModel):
    decision: Literal["ACCEPT", "REJECT"]
    action: Literal["INCREASE", "DECREASE"]
    severity: Literal["HIGH", "MEDIUM", "LOW", "ACCEPTABLE"]
    critique: str = Field(description="Feedback explicitly restating the step range for the gap.")

class AllReceiversOutput(BaseModel):
    reasoning: str
    receiver_critiques: List[SingleReceiverCritique]

class AggregatorOutput(BaseModel):
    reasoning: str
    aggregated_critique: str = Field(description="A unified summary explicitly stating which Transmitters should increase or decrease power based on the 5 receiver critiques.")

class ProposersFirstRound(BaseModel):
    reasoning: str
    powers: List[int] = Field(description="List of 5 integers representing P1 to P5, values 1 to 100.")

class ProposersRemainRounds(BaseModel):
    reasoning: str
    steps: List[int] = Field(description="List of 5 delta integers to add/subtract to P1-P5.")


# --- Node Definitions ---
def receivers(state: GraphState) -> GraphState:
    print(f"\n[Receivers Node] Iteration {state['iteration']}")
    
    interferences = []
    for rx in range(5):
        # Calculate received interference from all other Transmitters
        interf = sum(state['powers'][tx] * state['H_matrix'][tx][rx] for tx in range(5) if tx != rx)
        interferences.append(interf)
        
    state['interference_history'].append(interferences)
    
    prompt = f"""You represent 5 Receivers in a wireless network.
    You will receive the caused interference on your channels. 
    Threshold: {I_max}. Gap = interference - {I_max}.

    Follow these exact bands based on the Gap for EACH receiver:
    1. Gap > 1050: REJECT, DECREASE, HIGH.
    2. 500 <= Gap <= 1050: REJECT, DECREASE, MEDIUM.
    3. 10 <= Gap <= 499: REJECT, DECREASE, LOW.
    4. Gap <= -500: REJECT, INCREASE, HIGH.
    5. 0 < Gap <= 9: ACCEPT.
    6. -499 <= Gap <= 0: ACCEPT.

    Return JSON with 5 critiques, one for each receiver."""

    msg = "Current Interferences received:\n"
    for i in range(5):
        msg += f"Rx {i+1}: Interference = {interferences[i]}, Gap = {interferences[i] - I_max}\n"

    structured_critic = llm.with_structured_output(AllReceiversOutput)
    resp = structured_critic.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=msg)
    ])

    state['individual_critiques'] = [dict(c) for c in resp.receiver_critiques]
    state['decisions'] = [c.decision for c in resp.receiver_critiques]
    
    print(f"Decisions: {state['decisions']}")
    return state


def aggregator(state: GraphState) -> GraphState:
    print(f"[Aggregator Node] Iteration {state['iteration']}")
    
    prompt = """You are the Critique Aggregator.
    You will read 5 individual feedback critiques from 5 Receivers.
    Your job is to summarize this into ONE actionable paragraph. 
    Tell the 5 Transmitters (Tx1 to Tx5) explicitly who needs to INCREASE or DECREASE their power to help the network reach stability."""

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
    
    print(f"Aggregated Insight: {resp.aggregated_critique[:100]}...")
    return state


def build_train_prompt(train_examples) -> str:
    prompt_s = """You act as 5 Transmitters (Tx1 to Tx5).
    Propose an optimal power allocation list [P1, P2, P3, P4, P5] (values 1 to 100) based on the channel matrix H.
    
    Examples of good allocation:
    """
    # Keep context window safe by limiting to 10 examples
    for H, best_p in train_examples[:10]:
        prompt_s += f"Channel Matrix H:\n{H}\nOptimal Power Array: {best_p}\n\n"
    return prompt_s


def proposers(state: GraphState) -> GraphState:
    print(f"[Proposers Node] Iteration {state['iteration']}")

    if not state['aggregated_critique']:
        # Round 0: Use Context/Training Examples
        prompt = prmpt_train
        structured_critic = llm.with_structured_output(ProposersFirstRound)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Current Channel Matrix H:\n{state['H_matrix']}\nPropose Initial P1 to P5:")
        ])

        P_new = [int(max(1, min(100, p))) for p in resp.powers]
        state['powers'] = P_new
        state['allocation_history'].append(P_new)
        print(f"Initial Proposed Powers: {P_new}")

    else:
        # Round > 0: Use Aggregator's Feedback
        prompt = """You act as 5 Transmitters. Adjust your powers based on the Aggregated Critique.
        Severity-to-step guide: HIGH=20-30, MEDIUM=10-20, LOW=1-10.
        Provide a list of 5 steps to add/subtract. Make sure not to repeat steps if the situation stalled."""
        
        structured_critic = llm.with_structured_output(ProposersRemainRounds)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"""
            Current Powers: {state['powers']}
            Power History: {state['allocation_history']}
            Aggregated Critique: {state['aggregated_critique']}
            """)
        ])

        steps = resp.steps
        state['delta_hist'].append(steps)
        P_new = [int(max(1, min(100, state['powers'][i] + steps[i]))) for i in range(5)]
        state['powers'] = P_new
        state['allocation_history'].append(P_new)
        
        print(f"Steps taken: {steps}")
        print(f"New Powers: {P_new}")

    return state


def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
    print("Finalizer...\n")
    if state["iteration"] > state["max_iter"]:
        return "finalize"

    if "REJECT" in state['decisions']:
        return "revise"

    return "finalize"


# --- Graph Construction ---
workflow = StateGraph(GraphState)

workflow.add_node("Proposers", proposers)
workflow.add_node("Receivers", receivers)
workflow.add_node("Aggregator", aggregator)

workflow.set_entry_point("Proposers")
workflow.add_edge("Proposers", "Receivers")
workflow.add_edge("Receivers", "Aggregator")

workflow.add_conditional_edges(
    "Aggregator",
    finalizer,
    {
        "revise": "Proposers",
        "finalize": END,
    }
)

app = workflow.compile()

# --- Execution & Evaluation ---
print("Generating Channels (Grid Search Optimal Baseline)...")
ch = gen_channels_5(300) # Generate 300 to use 100 for training, 50 for testing.
train_examples = ch[0:100]
test_examples = ch[200:250]

prmpt_train = build_train_prompt(train_examples)

actual_SE = []
predicted_SE = []
above_max_interference = []
mean_error_per_agent = 0
cause_interference_count = 0

print("\n--- Starting 5-Agent LLM Negotiation ---")

for H_matrix, analytical_best_p in test_examples:
    initial_state = {
        "H_matrix": H_matrix,
        "powers": [100]*5, # Gets overwritten in iteration 0
        "allocation_history": [],
        "interference_history": [],
        "delta_hist": [],
        "iteration": 0,
        "max_iter": 3,  # Slight bump since 5 agents take more steps to converge
        "individual_critiques": [],
        "aggregated_critique": "",
        "decisions": []
    }
    result = app.invoke(initial_state)

    # Compute SE for Analytical vs LLM
    se_true = 0
    se_pred = 0
    max_interf_pred = 0
    
    for rx in range(5):
        # Actual SE Calculate
        sig_true = analytical_best_p[rx] * H_matrix[rx][rx]
        int_true = sum(analytical_best_p[tx] * H_matrix[tx][rx] for tx in range(5) if tx != rx)
        se_true += np.log2(1 + sig_true / (1 + int_true))

        # Predicted SE Calculate
        sig_pred = result['powers'][rx] * H_matrix[rx][rx]
        int_pred = sum(result['powers'][tx] * H_matrix[tx][rx] for tx in range(5) if tx != rx)
        se_pred += np.log2(1 + sig_pred / (1 + int_pred))
        
        # Track max interference over all receivers
        if int_pred > max_interf_pred:
            max_interf_pred = int_pred

    actual_SE.append(se_true)
    predicted_SE.append(se_pred)
    above_max_interference.append(max_interf_pred)

    if max_interf_pred > I_max:
        cause_interference_count += 1
        
    # Average MAE across the 5 dimensions for this sample
    avg_error = sum(abs(result['powers'][i] - analytical_best_p[i]) for i in range(5)) / 5.0
    mean_error_per_agent += avg_error

    print(f"\nFinal Powers Assigned by LLM: {result['powers']}")
    print(f"Analytical Best Powers was:   {analytical_best_p}")
    print(f"Final Worst Interference:     {max_interf_pred:.2f} (Limit: {I_max})\n")
    print("-" * 50)

# --- Plotting Results ---
print(f'\nMean Power Allocation Error per Agent: {mean_error_per_agent/len(test_examples):.2f}')
print(f"Network samples exceeding Interference: {cause_interference_count/len(test_examples):.2%}")

fig, ax1 = plt.subplots(1, figsize=(15, 6))
ax1.plot(actual_SE, color='red', label='Actual Sum SE', marker='o', linestyle='None')
ax1.plot(predicted_SE, color='blue', label='Predicted Sum SE', marker='x', linestyle='None')
ax1.set_title('Actual vs. Predicted Sum Spectral Efficiency (5 Agents)')
ax1.set_xlabel('Data Point Index')
ax1.set_ylabel('Sum SE Value')
ax1.legend()
ax1.grid(True)
plt.tight_layout()
plt.savefig("5_Agent_Sum_SE_Result.png")
plt.show()

fig2, ax2 = plt.subplots(1, figsize=(15, 6))
step = 5
indices = np.arange(0, len(actual_SE), step)
actual_SE_10 = [actual_SE[i] for i in indices]
predicted_SE_10 = [predicted_SE[i] for i in indices]
ax2.plot(indices, actual_SE_10, color='red', label=f'Actual Sum SE (every {step} steps)', marker='o', linestyle='-')
ax2.plot(indices, predicted_SE_10, color='blue', label=f'Predicted Sum SE (every {step} steps)', marker='x', linestyle='-')
ax2.set_title(f'Actual vs. Predicted Sum SE (Every {step} Steps)')
ax2.set_xlabel('Data Point Index')
ax2.set_ylabel('Sum SE Value')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig("5_Agent_Sum_SE_each_step.png")
plt.show()

fig, ax3 = plt.subplots(1, figsize=(15, 6))
ax3.plot(above_max_interference, color='blue', label='Max Received Interference', marker='x', linestyle='None')
ax3.set_title('Worst-Case Interference vs Threshold')
ax3.set_xlabel('Data Point Index')
ax3.set_ylabel('Interference Value')
ax3.legend()
ax3.grid(True)
plt.axhline(y=I_max, color='red', linestyle='--', linewidth=2, label=f'Threshold = {I_max}')
plt.tight_layout()
plt.savefig("5_Agent_Interference.png")
plt.show()