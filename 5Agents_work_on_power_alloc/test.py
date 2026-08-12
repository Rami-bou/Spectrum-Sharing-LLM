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
from typing import List, Dict, Any, Optional, Literal, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
import math
from dotenv import load_dotenv
import csv
from datetime import datetime

# number of primary receivers
N = 3
# number of secondary receivers
M = 3

MCS = [
    (2.0, 15),
    (5.0, 30),
    (9.0, 45),
    (11.0, 60),
    (15.0, 90),
    (18.0, 120),
    (20.0, 150)
]

data = []

f = 2e9
c = 3e8
wave = c / f
scale_factor = 1e8

secondary_I_max = 3000
primary_I_max = 1000

primary_transmitter = [80, 80]
secondary_transmitter = [20, 20]

random.seed(11)

def get_mcs_threshold(sinr_db):
    """Finds the minimum required SINR (dB) for the current state."""
    target_th = -999
    for th, rate in MCS:
        if sinr_db >= th:
            target_th = th
        else:
            break
    return target_th

def allocate_p1_knapsack_optimal(budget, direct_h_primary):
    """
    Distributes allowed_p1 among primary receivers to maximize aggregate 
    discrete throughput using greedy Knapsack selection.
    """
    N = len(direct_h_primary)
    P1_dist = [0] * N
    budget_left = budget

    while budget_left > 0:
        best_eff = -1.0
        best_user = -1
        best_cost = 0

        for i in range(N):
            sinr_lin = (P1_dist[i] * direct_h_primary[i]) 
            sinr_db = 10 * math.log10(sinr_lin) if sinr_lin > 0 else -999.0

            curr_rate = 0
            next_th = None
            next_rate = 0

            for th, rate in MCS:
                if sinr_db >= th:
                    curr_rate = rate
                elif next_th is None:
                    next_th = th
                    next_rate = rate
                    break

            if next_th is not None:
                target_sinr_lin = 10 ** (next_th / 10.0)
                required_p1 = target_sinr_lin / direct_h_primary[i]
                cost = int(math.ceil(required_p1 - P1_dist[i]))

                if 0 < cost <= budget_left:
                    value = next_rate - curr_rate
                    eff = value / float(cost)

                    if eff > best_eff:
                        best_eff = eff
                        best_user = i
                        best_cost = cost

        if best_user != -1:
            P1_dist[best_user] += best_cost
            budget_left -= best_cost
        else:
            best_user = max(range(N), key=lambda k: direct_h_primary[k])
            P1_dist[best_user] += budget_left
            budget_left = 0

    return P1_dist

def allocate_p2_knapsack_optimal(allowed_p2, direct_h_secondary, cross_h_secondary, P1_dist):
    """
    Distributes allowed_p2 among secondary receivers to maximize aggregate 
    discrete throughput using greedy Knapsack selection.
    """
    M = len(direct_h_secondary)
    P2_dist = [0] * M
    budget = allowed_p2

    total_p1_interf = [sum(P1_dist) * cross_h_secondary[i] for i in range(M)]

    while budget > 0:
        best_eff = -1.0
        best_user = -1
        best_cost = 0

        for i in range(M):
            sinr_lin = (P2_dist[i] * direct_h_secondary[i]) / (1.0 + total_p1_interf[i])
            sinr_db = 10 * math.log10(sinr_lin) if sinr_lin > 0 else -999.0

            curr_rate = 0
            next_th = None
            next_rate = 0

            for th, rate in MCS:
                if sinr_db >= th:
                    curr_rate = rate
                elif next_th is None:
                    next_th = th
                    next_rate = rate
                    break

            if next_th is not None:
                target_sinr_lin = 10 ** (next_th / 10.0)
                required_p2 = (target_sinr_lin * (1.0 + total_p1_interf[i])) / direct_h_secondary[i]
                cost = int(math.ceil(required_p2 - P2_dist[i]))

                if 0 < cost <= budget:
                    value = next_rate - curr_rate
                    eff = value / float(cost)

                    if eff > best_eff:
                        best_eff = eff
                        best_user = i
                        best_cost = cost

        if best_user != -1:
            P2_dist[best_user] += best_cost
            budget -= best_cost
        else:
            best_user = max(range(M), key=lambda k: direct_h_secondary[k])
            P2_dist[best_user] += budget
            budget = 0

    return P2_dist

def get_discrete_rate(sinr_linear):
    """Converts linear SINR to dB and maps it to a discrete data rate."""
    if sinr_linear <= 0:
        return 0
    
    sinr_db = 10 * math.log10(sinr_linear)
    
    achieved_rate = 0
    for threshold, rate in MCS:
        if sinr_db >= threshold:
            achieved_rate = rate
        else:
            break
            
    return achieved_rate

def calculate_primary_discrete_rate(P1_vector, P2_vector, direct_h_primary, cross_h_primary):
    """Calculates total Primary Throughput based on discrete MCS levels."""
    total_throughput_mbps = 0
    total_P2 = sum(P2_vector)
    
    for j in range(len(P1_vector)):
        signal = P1_vector[j] * direct_h_primary[j]
        interference_from_secondary = total_P2 * cross_h_primary[j]
        
        sinr_linear = signal / (1.0 + interference_from_secondary)
        total_throughput_mbps += get_discrete_rate(sinr_linear)
        
    return total_throughput_mbps

def calculate_secondary_discrete_rate(P1_vector, P2_vector, direct_h_secondary, cross_h_secondary):
    """Calculates total Secondary Throughput based on discrete MCS levels."""
    total_throughput_mbps = 0
    total_P1 = sum(P1_vector) 
    
    for j in range(len(P2_vector)):
        signal = P2_vector[j] * direct_h_secondary[j]
        interference_from_primary = total_P1 * cross_h_secondary[j]
        
        sinr_linear = signal / (1.0 + interference_from_primary)
        total_throughput_mbps += get_discrete_rate(sinr_linear)
        
    return total_throughput_mbps

def gen_channels(length):
    while len(data) < length:
        position_primary_receiver = []
        direct_h_primary = []
        for i in range(N):
            dist_r = random.uniform(10, 35)
            angle = random.uniform(0, 2 * math.pi)
            rp = [primary_transmitter[0] + dist_r * math.cos(angle),
                  primary_transmitter[1] + dist_r * math.sin(angle)]
            position_primary_receiver.append(rp)
            d = np.sqrt((rp[0]-primary_transmitter[0])**2 + (rp[1]-primary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            direct_h_primary.append(int(round(h * scale_factor, 2)))

        position_secondary_receiver = []
        direct_h_secondary = []
        for i in range(M):
            dist_r = random.uniform(1, 5)
            angle = random.uniform(0, 2 * math.pi)
            rs = [secondary_transmitter[0] + dist_r * math.cos(angle),
                  secondary_transmitter[1] + dist_r * math.sin(angle)]
            position_secondary_receiver.append(rs)
            d = np.sqrt((rs[0]-secondary_transmitter[0])**2 + (rs[1]-secondary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            direct_h_secondary.append(int(round(h * scale_factor, 2)))

        cross_h_primary = []
        for pos in position_primary_receiver:
            d = np.sqrt((pos[0]-secondary_transmitter[0])**2 + (pos[1]-secondary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            cross_h_primary.append(int(round(h * scale_factor, 2)))

        cross_h_secondary = []
        for pos in position_secondary_receiver:
            d = np.sqrt((pos[0]-primary_transmitter[0])**2 + (pos[1]-primary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            cross_h_secondary.append(int(round(h * scale_factor, 2)))

        allowed_p1 = int(round(secondary_I_max / max(cross_h_secondary)))
        if allowed_p1 < N:
            continue

        sum_inverses = sum(1.0 / v for v in direct_h_primary)
        aggregate_signal = allowed_p1 / sum_inverses

        if aggregate_signal <= 0 or get_mcs_threshold(10 * math.log10(aggregate_signal)) < 0:
            continue

        baseline_sinr_db = 10 * math.log10(aggregate_signal)
        target_th = get_mcs_threshold(baseline_sinr_db)
        min_linear_sinr = 10 ** (target_th / 10.0)
        max_interference = (aggregate_signal / min_linear_sinr) - 1.0

        if max_interference <= 0:
            continue

        p2_limits = []
        for cross_h in cross_h_primary:
            if cross_h > 0:
                p2_limits.append(max_interference / cross_h)

        if not p2_limits:
            continue

        allowed_p2 = int(math.floor(min(p2_limits)))
        if allowed_p2 < M:
            continue

        data.append([direct_h_primary, direct_h_secondary, cross_h_primary, cross_h_secondary, allowed_p1, allowed_p2])

    return data

llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)

class GraphState(TypedDict):
    direct_primary_channels: List[int]
    direct_secondary_channels: List[int]
    cross_primary_channels: List[int]
    cross_secondary_channels: List[int]

    P1: List[int]
    P2: List[int]

    primary_critique: str
    secondary_critique: str

    primary_decision: str

    delta_hist: List[int]

    iteration: int

class PrimaryOutput(BaseModel):
    decision: Literal["ACCEPT", "REJECT"] = Field(description="The final decision based strictly on the rules.")
    action: Literal["INCREASE", "DECREASE"] = Field(description="Should the target power be increased or decreased?")
    severity: Literal["HIGH", "MEDIUM", "LOW"] = Field(description="Magnitude of correction needed, independent of direction.")
    critique: str = Field(description="Explicit instructions detailing what to do with the target array.")

def primary(state: GraphState) -> GraphState:
    """
    The primary transmitter evaluates the secondary's proposed power allocation (P2) and provides feedback.
    """
    total_p2 = sum(state['P2'])
    margins = []
    
    for j in range(len(state['P1'])):
        signal = state['P1'][j] * state['direct_primary_channels'][j]
        if signal <= 0:
            continue
            
        baseline_sinr_db = 10 * math.log10(signal)
        target_th = get_mcs_threshold(baseline_sinr_db)
        if target_th < 0:
            continue
            
        interference = total_p2 * state['cross_primary_channels'][j]
        actual_sinr_linear = signal / (1.0 + interference)
        actual_sinr_db = 10 * math.log10(actual_sinr_linear) if actual_sinr_linear > 0 else -999
        
        margin = actual_sinr_db - target_th
        margins.append(margin)

    worst_margin = min(margins) if margins else -999.0
    print(f"\n[Primary Evaluator] Worst MCS Margin: {worst_margin:.2f} dB")

    prompt_primary = f"""You are the Central Network Evaluator protecting Primary users' discrete data rates.
    You evaluate the 'Worst MCS Margin' (measured in dB). 
    
    Follow these exact decision bands:
    1. Margin < -3.0 dB: EMERGENCY, severe rate loss. decision=REJECT, action=DECREASE, severity=HIGH.
    2. -3.0 dB <= Margin < -0.5 dB: action=DECREASE, severity=MEDIUM.
    3. -0.5 dB <= Margin < 0.0 dB: decision=REJECT, action=DECREASE, severity=LOW.
    4. 0.0 dB <= Margin <= 2.0 dB: decision=ACCEPT.
    5. 2.0 dB < Margin <= 5.0 dB: decision=REJECT, action=INCREASE, severity=LOW.
    6. Margin > 5.0 dB: Far below capacity, secondary is being overly conservative. decision=REJECT, action=INCREASE, severity=HIGH.

    Your critique must explicitly mention the amount of the worst margin.

    Return JSON matching the schema.
    """

    structured_critic = llm.with_structured_output(PrimaryOutput)
    resp = structured_critic.invoke([
        SystemMessage(content=prompt_primary),
        HumanMessage(content=f"""
        Total P2 Proposed: {total_p2}
        Worst Primary MCS Margin: {worst_margin:.2f} dB
        """
        )
    ])

    state['primary_critique'] = resp.critique
    state['primary_decision'] = resp.decision
    state['iteration'] += 1
    
    print(f"[Decision]: {resp.decision} ({resp.severity})")
    print(f"[Critique]: {resp.critique}")

    return state

# --- NEW: Changed list output to an integer sum output ---
class SecondaryOutput(BaseModel):
    reasoning: str = Field(description="You provide a brief reasoning before making any decision, expalaining why you will do this.")
    total_secondary_power: int = Field(description="Your predicted total power budget (sum of P2) for all secondary receivers.")

class SecondaryRemainRounds(BaseModel):
    reasoning: str = Field(description="You provide a brief reasoning before making any decision, expalaining why you will do this.")
    step: int = Field(description="The step to add/substract you think that i will hit the best P2.")

def secondary(state:GraphState) -> GraphState:
    """The Secondary Network Optimizer"""
    if not state['primary_critique']:
        structured_critic = llm.with_structured_output(SecondaryOutput)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt_secondary_allocation),
            HumanMessage(content=f"""Predict the Total Power (P2 budget) for the secondary network:

            If the direct secondary channels are {state['direct_secondary_channels']}
            And the cross secondary channels are {state['cross_secondary_channels']}
            Then the Total Power (P2) is:  
            """
            )
        ])

        print(f"P2 First Round Total Allocation: {resp.total_secondary_power}")
        # --- NEW: Immediately distribute the sum using Knapsack ---
        state['P2'] = allocate_p2_knapsack_optimal(
            resp.total_secondary_power, 
            state['direct_secondary_channels'], 
            state['cross_secondary_channels'], 
            state['P1']
        )
        print(f"Distributed as: {state['P2']}")

    else:
        prompt = f"""You are the Secondary Network Optimizer operating alongside a Primary Network.
        Your goal is to adjust the total Secondary Power (P2) budget based on the Primary Evaluator's critique.
        
        The Primary Evaluator monitors the 'Worst MCS Margin' (in dB). The sweet spot is a margin exactly between 0.0 dB and 2.0 dB.
        
        You must select an integer `step` to adjust your total P2 budget strictly from the allowed lists below. 
        
        [DECREASE ACTIONS - Margin < 0.0 dB]
        Allowed Steps: [-30, -25, -20, -15, -10, -5, -3, -2, -1]
        
        [INCREASE ACTIONS - Margin > 2.0 dB]
        Allowed Steps: [+1, +2, +3, +5, +10, +15]

        CRITICAL RULES:
        1. You must ONLY select a step value from the Allowed Steps lists provided above.
        2. Always output a NEGATIVE integer if the action is DECREASE. Always output a POSITIVE integer if the action is INCREASE.
        3. Oscillation Check: Look at your `delta_hist`. If your last step caused the margin to flip polarity, reverse direction and pick a smaller step.

        Return JSON matching the schema.
        """

        structured_critic = llm.with_structured_output(SecondaryRemainRounds)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"""
            Secondary Current Total P2 Proposal: {sum(state['P2'])}
            Your own step history: {state['delta_hist']}
            Primary decision: {state['primary_decision']}
            Primary critique: {state["primary_critique"]}
            """
            )
        ])

        total_p2 = sum(state['P2'])
        state['delta_hist'].append(resp.step)

        print(f"Delta: {resp.step}")

        P2_new = int(max(1, total_p2 + resp.step))
        # --- NEW: Continue distributing updated sum using Knapsack ---
        state['P2'] = allocate_p2_knapsack_optimal(
            P2_new, 
            state['direct_secondary_channels'], 
            state['cross_secondary_channels'], 
            state['P1']
        )

        print(f"New total power after delta: {P2_new} -> Distributed: {state['P2']}")

    return state

# --- NEW: Updated prompt builder to train LLM on Sums, not arrays ---
def build_prompt(train):
    prompt_primary = f"""You are the secondary transmitter in a wireless communication scenario.
    Your job is to predict the optimal TOTAL transmission power (P2 budget) for your network.
    Here are some examples of perfect total allocations based on channel states:\n
    """
    for i in range(len(train)):
        prompt_primary += f"""
        If the direct secondary channels are {train[i][1]} and cross secondary channels are {train[i][3]}
        Then the Optimal Total Power (P2) is: {train[i][5]}    
        """
    
    prompt_primary += "\nReturn JSON matching the schema."

    return prompt_primary

def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
    print("Finalizer...\n")
    if state["iteration"] > 3:
        return "finalize"
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

data = gen_channels(190)
train = data[:90]
test = data[90:150]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
prompt_secondary_allocation = build_prompt(train)

RESULT_DIR = os.path.join("results", f"baseline_{timestamp}")
os.makedirs(RESULT_DIR, exist_ok=True)

def save_file(path):
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BASELINE PERFORMANCE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Average Secondary Rate      : {np.mean(se_pred_list):.2f} Mbps\n")
        f.write(f"Average Optimal Rate        : {np.mean(se_true_list):.2f} Mbps\n")
        efficiency = 100 * np.mean(se_pred_list) / np.mean(se_true_list)
        f.write(f"Efficiency                 : {efficiency:.2f} %\n\n")
        f.write(f"Average Interference        : {np.mean(interf_pred_list):.2f}\n")
        f.write(f"Maximum Interference        : {np.max(interf_pred_list):.2f}\n")
        violation_rate = 100 * np.mean(violation_list)
        f.write(f"Violation Rate             : {violation_rate:.2f} %\n\n")
        success_rate = 100 * np.mean(success_list)
        f.write(f"Negotiation Success Rate   : {success_rate:.2f} %\n")

se_pred_list = []
se_true_list = []
interf_pred_list = []
interf_true_list = []
success_list = []
violation_list = []
se_pred_list_primary = []
se_true_list_primary = []

csv_path = os.path.join(RESULT_DIR, "benchmark.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Sample",
    "TrueRate",
    "PredRate",
    "TrueInterference",
    "PredInterference",
    "Violation",
    "Rounds",
    "Decision",
    "TrueP2",
    "PredP2"
])

print(f"\nStarting Benchmark over {len(test)} Test Samples...")

for i in range(len(test)):
    direct_h_prim = test[i][0]
    direct_h_sec = test[i][1]
    cross_h_prim = test[i][2]
    cross_h_sec = test[i][3]

    true_p1_total = test[i][4]
    true_p2_total = test[i][5]

    # --- NEW: Calculate the true array distributions to pass into GraphState ---
    true_p1_dist = allocate_p1_knapsack_optimal(true_p1_total, direct_h_prim)
    true_p2_dist = allocate_p2_knapsack_optimal(true_p2_total, direct_h_sec, cross_h_sec, true_p1_dist)

    initial_state = {
        "direct_primary_channels": direct_h_prim,
        "direct_secondary_channels": direct_h_sec,
        "cross_primary_channels": cross_h_prim,
        "cross_secondary_channels": cross_h_sec,
        "P1": true_p1_dist,  # Now passing the Distributed Array
        "P2": [0] * M,
        "primary_critique": "",
        "secondary_critique": "",
        "primary_decision": "",
        "delta_hist": [],
        "iteration": 0
    }

    result = app.invoke(initial_state)
    pred_p2_dist = result['P2']

    rate_pred = calculate_secondary_discrete_rate(true_p1_dist, pred_p2_dist, direct_h_sec, cross_h_sec)
    rate_true = calculate_secondary_discrete_rate(true_p1_dist, true_p2_dist, direct_h_sec, cross_h_sec)
    se_pred_list.append(rate_pred)
    se_true_list.append(rate_true)

    rate_pred_primary = calculate_primary_discrete_rate(true_p1_dist, pred_p2_dist, direct_h_prim, cross_h_prim)
    rate_true_primary = calculate_primary_discrete_rate(true_p1_dist, true_p2_dist, direct_h_prim, cross_h_prim)
    se_pred_list_primary.append(rate_pred_primary)
    se_true_list_primary.append(rate_true_primary)

    max_interf_pred = sum(pred_p2_dist) * max(cross_h_prim)
    max_interf_true = sum(true_p2_dist) * max(cross_h_prim)
    interf_pred_list.append(max_interf_pred)
    interf_true_list.append(max_interf_true)
    
    success_list.append(1 if result["primary_decision"] == "ACCEPT" else 0)
    violation_list.append(1 if max_interf_pred > primary_I_max else 0)

    csv_writer.writerow([
        i + 1,
        rate_true,
        rate_pred,
        max_interf_true,
        max_interf_pred,
        violation_list[i],
        result["iteration"],
        result["primary_decision"],
        sum(true_p2_dist),
        sum(pred_p2_dist)
    ])

    print(f"Sample {i+1}/100 | True Rate: {rate_true} | Pred Rate: {rate_pred} | Pred Interf: {max_interf_pred:.1f}")
    print(f"True Total P2: {sum(true_p2_dist)} -> Dist: {true_p2_dist}")
    print(f"Pred Total P2: {sum(pred_p2_dist)} -> Dist: {result['P2']}")

csv_file.close()
metrics_path = os.path.join(RESULT_DIR, "metrics.txt")
save_file(metrics_path)
print(f"System Benchmark Before attack:\n")
print(f"Average Secondary Rate (True): {np.mean(se_true_list):.2f}")
print(f"Average Secondary Rate (Predicted): {np.mean(se_pred_list):.2f}")
print(f"Average Interference (Predicted): {np.mean(interf_pred_list):.2f}")
print(f"Max Interference (Predicted): {np.max(interf_pred_list):.2f}")
print(f"Efficiency: {np.mean(success_list):.0%}")
print(f"Constraint Violations: {np.sum(violation_list):.0%}")

if len(test) > 0:
    bin_size = 5
    num_bins = (len(test) + bin_size - 1) // bin_size
    bin_x = [i * bin_size for i in range(1, num_bins + 1)]

    binned_se_pred = [np.mean(se_pred_list[i : i + bin_size]) for i in range(0, len(se_pred_list), bin_size)]
    binned_se_true = [np.mean(se_true_list[i : i + bin_size]) for i in range(0, len(se_true_list), bin_size)]
    binned_se_pred_primary = [np.mean(se_pred_list_primary[i : i + bin_size]) for i in range(0, len(se_pred_list_primary), bin_size)]
    binned_se_true_primary = [np.mean(se_true_list_primary[i : i + bin_size]) for i in range(0, len(se_true_list_primary), bin_size)]
    binned_interf_pred = [np.mean(interf_pred_list[i : i + bin_size]) for i in range(0, len(interf_pred_list), bin_size)]
    binned_interf_true = [np.mean(interf_true_list[i : i + bin_size]) for i in range(0, len(interf_true_list), bin_size)]

    plt.figure(figsize=(10, 5))
    plt.plot(bin_x, binned_se_true, label='True Optimal Secondary Rate', color='blue', linestyle='--', marker='o', linewidth=2)
    plt.plot(bin_x, binned_se_pred, label='LLM Agent Secondary Rate', color='red', linestyle='-', marker='s', linewidth=2)
    plt.title('Secondary Network Sum Rate (Averaged Every 5 Test Samples)', fontsize=13)
    plt.xlabel('Test Sample Index (Bin Size = 5)', fontsize=11)
    plt.ylabel('Average Secondary Rate (Mbps)', fontsize=11)
    plt.xticks(bin_x)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "secondary_rate_normal.png"), dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 5))
    plt.plot(bin_x, binned_se_true_primary, label='True Optimal Primary Rate', color='blue', linestyle='--', marker='o', linewidth=2)
    plt.plot(bin_x, binned_se_pred_primary, label='LLM Agent Primary Rate', color='red', linestyle='-', marker='s', linewidth=2)
    plt.title('Primary Network Sum Rate (Averaged Every 5 Test Samples)', fontsize=13)
    plt.xlabel('Test Sample Index (Bin Size = 5)', fontsize=11)
    plt.ylabel('Average Primary Rate (Mbps)', fontsize=11)
    plt.xticks(bin_x)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "primary_rate_normal.png"), dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 5))
    plt.axhline(y=primary_I_max, color='black', linestyle='-', linewidth=2, label=f'Primary Interference Limit (${{I_{{max}}}}={primary_I_max}$)')
    plt.plot(bin_x, binned_interf_true, label='True Optimal Interference', color='blue', linestyle='--', marker='o', linewidth=2)
    plt.plot(bin_x, binned_interf_pred, label='LLM Agent Interference', color='red', linestyle='-', marker='x', linewidth=2, markersize=8)
    plt.title('Primary Network Protection: Caused Interference (Averaged Every 5 Test Samples)', fontsize=13)
    plt.xlabel('Test Sample Index (Bin Size = 5)', fontsize=11)
    plt.ylabel('Average Max Interference Injected', fontsize=11)
    plt.xticks(bin_x)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "primary_interference_normal.png"), dpi=300, bbox_inches="tight")

    plt.show()
else:
    print("No test samples to plot.")