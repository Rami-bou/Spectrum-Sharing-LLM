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

N = 3
M = 3

data = []

f = 2e9
c = 3e8
wave = c / f
scale_factor = 1e8

secondary_I_max = 3000
primary_I_max = 1000
random.seed(10)

MCS = [
    (2.0, 15),
    (5.0, 30),
    (9.0, 45),
    (11.0, 60),
    (15.0, 90),
    (18.0, 120),
    (20.0, 150)
]

def get_mcs_threshold(sinr_db):
    """Finds the minimum required SINR (dB) for the current state."""
    target_th = -999
    for th, rate in MCS:
        if sinr_db >= th:
            target_th = th
        else:
            break
    return target_th

def allocate_p2_knapsack_optimal(allowed_p2, direct_h_secondary, cross_h_secondary, P1_dist):
    """
    Distributes allowed_p2 among secondary receivers to maximize aggregate 
    discrete throughput using greedy Knapsack selection.
    """
    M = len(direct_h_secondary)
    P2_dist = [0] * M
    budget = allowed_p2

    # Pre-calculate interference caused by Primary onto each Secondary receiver
    total_p1_interf = [sum(P1_dist) * cross_h_secondary[i] for i in range(M)]

    # Greedy allocation loop
    while budget > 0:
        best_eff = -1.0
        best_user = -1
        best_cost = 0

        for i in range(M):
            # Calculate current SINR in dB
            sinr_lin = (P2_dist[i] * direct_h_secondary[i]) / (1.0 + total_p1_interf[i])
            sinr_db = 10 * math.log10(sinr_lin) if sinr_lin > 0 else -999.0

            # Find current rate and next MCS threshold
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

            # If an upgrade tier exists, evaluate cost and efficiency
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

        # If a valid upgrade user was found, purchase the upgrade
        if best_user != -1:
            P2_dist[best_user] += best_cost
            budget -= best_cost
        else:
            # If remaining budget cannot push ANY user to a higher MCS level,
            # dump the leftover budget into the receiver with the strongest direct channel.
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
        
        # Calculate physical linear SINR
        sinr_linear = signal / (1.0 + interference_from_secondary)
        
        # Map to discrete hardware throughput
        total_throughput_mbps += get_discrete_rate(sinr_linear)
        
    return total_throughput_mbps

def gen_channels(length):
    while len(data) < length:
        primary_transmitter = [50, 50]
        secondary_transmitter = [30, 30]

        # 1. Primary Channels
        position_primary_receiver = []
        direct_h_primary = []
        for i in range(N):
            rp = [random.uniform(10, 90), random.uniform(10, 90)]
            position_primary_receiver.append(rp)
            d = np.sqrt((rp[0]-primary_transmitter[0])**2 + (rp[1]-primary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            direct_h_primary.append(int(round(h * scale_factor, 2)))

        # 2. Secondary Channels
        position_secondary_receiver = []
        direct_h_secondary = []
        for i in range(M):
            rs = [random.uniform(10, 50), random.uniform(10, 50)]
            position_secondary_receiver.append(rs)
            d = np.sqrt((rs[0]-secondary_transmitter[0])**2 + (rs[1]-secondary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            direct_h_secondary.append(int(round(h * scale_factor, 2)))

        # 3. Cross Channels
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

        # 4. P1 Power Distribution
        allowed_p1 = int(round(secondary_I_max / max(cross_h_secondary)))
        if allowed_p1 < N:
            continue

        inverses = [1.0 / v for v in direct_h_primary]
        sum_inverses = sum(inverses)
        P1_dist = [int(round((inv / sum_inverses) * allowed_p1)) for inv in inverses]

        # 5. Calculate Max Allowed P2 based on Primary MCS Cliffs
        p2_limits = []
        for j in range(N):
            signal = P1_dist[j] * direct_h_primary[j]
            if signal <= 0:
                continue
            
            baseline_sinr_db = 10 * math.log10(signal)
            target_th = get_mcs_threshold(baseline_sinr_db)
            if target_th < 0:
                continue
            
            min_linear_sinr = 10 ** (target_th / 10.0)
            max_interference = (signal / min_linear_sinr) - 1.0
            
            if max_interference > 0 and cross_h_primary[j] > 0:
                p2_limits.append(max_interference / cross_h_primary[j])

        if not p2_limits:
            continue

        allowed_p2 = int(math.floor(min(p2_limits)))
        if allowed_p2 < M:
            continue

        inverses = [1.0 / v for v in direct_h_secondary]
        sum_inverses = sum(inverses)
        P2_dist = [int(round((inv / sum_inverses) * allowed_p2)) for inv in inverses]
        
        data.append([direct_h_primary, direct_h_secondary, cross_h_primary, cross_h_secondary, P1_dist, P2_dist])

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
    The primary transmitter evaluates the secondary's proposed power allocation (P2) and provides feedback based on 
    the worst-case MCS margin across all primary receivers.
    """
    total_p2 = sum(state['P2'])
    margins = []
    
    # 1. Calculate the MCS margin for every primary receiver
    for j in range(len(state['P1'])):
        signal = state['P1'][j] * state['direct_primary_channels'][j]
        if signal <= 0:
            continue
            
        # Baseline SINR in dB (when P2 = 0)
        baseline_sinr_db = 10 * math.log10(signal)
        
        # Target MCS cliff threshold for this receiver
        target_th = get_mcs_threshold(baseline_sinr_db)
        if target_th < 0:
            continue
            
        # Actual SINR in dB with current P2 proposal
        interference = total_p2 * state['cross_primary_channels'][j]
        actual_sinr_linear = signal / (1.0 + interference)
        actual_sinr_db = 10 * math.log10(actual_sinr_linear) if actual_sinr_linear > 0 else -999
        
        # Margin: How far above/below the cliff edge are we?
        margin = actual_sinr_db - target_th
        margins.append(margin)

    # 2. Network safety depends on the weakest receiver
    worst_margin = min(margins) if margins else -999.0
    print(f"\n[Primary Evaluator] Worst MCS Margin: {worst_margin:.2f} dB")

    prompt_primary = f"""You are the Central Network Evaluator protecting Primary users' discrete data rates.
    You evaluate the 'Worst MCS Margin' (measured in dB). 
    - A positive Margin means secondary interference is safely absorbed within the MCS step (no data loss).
    - A negative Margin means secondary interference pushed a primary user off their MCS cliff, causing rate loss.
    
    Follow these exact decision bands:
    1. Margin < -3.0 dB: EMERGENCY, severe rate loss. decision=REJECT, action=DECREASE, severity=HIGH.
    2. -3.0 dB <= Margin < -0.5 dB: Noticeable rate drop. decision=REJECT, action=DECREASE, severity=MEDIUM.
    3. -0.5 dB <= Margin < 0.0 dB: Just barely pushed over the cliff edge. decision=REJECT, action=DECREASE, severity=LOW.
    4. 0.0 dB <= Margin <= 2.0 dB: OPTIMAL COOPERATION! Right on the cliff edge with zero rate loss. decision=ACCEPT, severity=LOW.
    5. 2.0 dB < Margin <= 4.0 dB: Below capacity, wasting secondary power budget. decision=REJECT, action=INCREASE, severity=LOW.
    6. Margin > 4.0 dB: Far below capacity, secondary is being overly conservative. decision=REJECT, action=INCREASE, severity=HIGH.

    Your critique must explicitly mention the numeric step range for the matched band so the secondary user knows how to adjust.

    Return JSON matching the schema.
    """

    structured_critic = llm.with_structured_output(PrimaryOutput)
    resp = structured_critic.invoke([
        SystemMessage(content=prompt_primary),
        HumanMessage(content=f"""
        P2 Allocations proposed: {state['P2']}
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

class SecondaryOutput(BaseModel):
    reasoning: str = Field(description="You provide a brief reasoning before making any decision, expalaining why you will do this.")
    allocation_secondary: List[int] = Field(description="Your allocation for all of your secondary receivers.")

class SecondaryRemainRounds(BaseModel):
    reasoning: str = Field(description="You provide a brief reasoning before making any decision, expalaining why you will do this.")
    step: int = Field(description="The step to add/substract you think that i will hit the best P2.")

def secondary(state:GraphState) -> GraphState:
    """The Secondary Network Optimizer, operating alongside a Primary Network, 
    aims to maximize the Secondary Power (P2) budget without violating the Primary user's discrete MCS data rate.
    """
    if not state['primary_critique']:
        structured_critic = llm.with_structured_output(SecondaryOutput)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt_secondary_allocation),
            HumanMessage(content=f"""Complete thr following allocations depends on the channels:

            If the secondary channels are {state['direct_secondary_channels']}
            Then the Power (P2) allocation are:  
            """
            )
        ])

        print(f"P2 First Round Allocation {resp.allocation_secondary}")
        state['P2'] = resp.allocation_secondary

    else:
        prompt = f"""You are the Secondary Network Optimizer operating alongside a Primary Network.
        Your goal is to find the maximum possible Secondary Power (P2) budget without violating the Primary user's discrete MCS data rate. 
        
        The Primary Evaluator monitors the 'Worst MCS Margin' (in dB). The sweet spot is a margin exactly between 0.0 dB and 2.0 dB. 
        Based on the Primary's critique, you must output an integer `step` to adjust your total P2 power budget.
        
        Use this exact mapping to determine your step size based on the Primary's Margin and Severity:
        
        [DECREASE ACTIONS - Negative Step Values]
        - Severity HIGH (Margin < -3.0 dB): EMERGENCY. You completely jammed the Primary user. 
          Action: Output a large negative step (e.g., -20 to -40).
        - Severity MEDIUM (-3.0 to -0.5 dB): Noticeable rate drop. 
          Action: Output a moderate negative step (e.g., -10 to -15).
        - Severity LOW (-0.5 to 0.0 dB): Just barely pushed over the cliff edge. 
          Action: Output a tiny negative step (e.g., -1 to -5).
          
        [INCREASE ACTIONS - Positive Step Values]
        - Severity LOW (2.0 to 4.0 dB): The primary is safe, and you have a small amount of excess room. 
          Action: Output a small positive step (e.g., +5 to +10).
        - Severity HIGH (Margin > 4.0 dB): The primary has a massive excess margin. You are leaving free throughput on the table. 
          Action: Output a large positive step (e.g., +20 to +40).

        CRITICAL RULES:
        1. Always output a NEGATIVE integer if the action is DECREASE.
        2. Always output a POSITIVE integer if the action is INCREASE.
        3. Review your `delta_hist` to avoid repeating the exact same failed step size. If you are bouncing back and forth over the cliff, cut your step size in half.

        Return JSON matching the schema.
        """

        structured_critic = llm.with_structured_output(SecondaryRemainRounds)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"""
            Secondary Current P2 Proposal: {state['P2']}
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
        inverses = [1.0 / v for v in state['direct_secondary_channels']]
        sum_inverses = sum(inverses)
        state['P2'] = [int(round((inv / sum_inverses) * P2_new)) for inv in inverses]

        print(f"New power after delta: {state['P2']}")

    return state

def build_prompt(train):
    prompt_primary = f"""You are the secondary transmitter in a wireless communication scenario.
    Your job is to allocate a transmission power for each one of your receivers.
    Here is some examples on good allocations based on the channel states:\n
    """
    for i in range(len(train)):
        prompt_primary += f"""
        If the secondary channels are {train[i][1]}
        Then the Power (P2) allocation are: {train[i][5]}    
        """
    
    prompt_primary += "\nReturn JSON matching the schema."

    return prompt_primary

def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
    print("Finalizer...\n")
    if state["iteration"] > 3:
        return "finalize"
    # earsly stop
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

data = gen_channels(120)
train = data[:90]
test = data[90:100]
prompt_secondary_allocation = build_prompt(train)
all_pred_P2 = []
all_true_P2 = []
se_pred_list = []
se_true_list = []

print("\nStarting Benchmark over Test Dataset...")
for i in range(len(test)):
    direct_h_pri = test[i][0] 
    cross_h_pri = test[i][2]  
    true_p1 = test[i][4]      
    true_p2 = test[i][5] 
    
    initial_state = {
        "direct_primary_channels": test[i][0],
        "direct_secondary_channels": test[i][1],
        "cross_primary_channels": test[i][2],
        "cross_secondary_channels": test[i][3],
        "P1": test[i][4],                      
        "P2": [0] * M,
        "primary_critique": "",
        "primary_decision": "",
        "delta_hist": [],
        "iteration": 0
    }

    result = app.invoke(initial_state)
    
    pred_p2 = result['P2']
    
    all_pred_P2.append(pred_p2)
    all_true_P2.append(true_p2)
    
    se_pred_list.append(calculate_primary_discrete_rate(true_p1, pred_p2, direct_h_pri, cross_h_pri))
    se_true_list.append(calculate_primary_discrete_rate(true_p1, true_p2, direct_h_pri, cross_h_pri))

    print(f"Allocation P2 pred: {result['P2']}")
    print(f"Allocation P2 true: {test[i][5]}")

all_pred_P2 = np.array(all_pred_P2) 
all_true_P2 = np.array(all_true_P2) 

mae_per_receiver = np.mean(np.abs(all_pred_P2 - all_true_P2), axis=0)

print("\n" + "="*40)
print(" BENCHMARK RESULTS: MEAN ABSOLUTE ERROR ")
print("="*40)
for j in range(len(mae_per_receiver)):
    print(f"Secondary Receiver {j+1} MAE: {mae_per_receiver[j]:.2f} Watts")
print("="*40 + "\n")

# --- REPLACEMENT PLOTTING SECTION ---
window_size = 5 if len(test) < 50 else 10

def moving_average(data, w):
    """Calculates the moving average shifting by 1 step at a time."""
    return np.convolve(data, np.ones(w), 'valid') / w

# Sum the total Watts assigned to the Secondary Network for each test sample
p2_pred_sum = [sum(p) for p in all_pred_P2]
p2_true_sum = [sum(p) for p in all_true_P2]

smoothed_p2_pred = moving_average(p2_pred_sum, window_size)
smoothed_p2_true = moving_average(p2_true_sum, window_size)

plt.figure(figsize=(12, 6))
# Plot True Optimal first (thicker blue line)
plt.plot(smoothed_p2_true, label='True Optimal Secondary Power (Watts)', color='blue', linestyle='--', marker='o', markersize=6, linewidth=2)
# Plot Agent on top (slightly thinner red line so you can see both)
plt.plot(smoothed_p2_pred, label='Agent-Allocated Secondary Power (Watts)', color='red', linestyle='-', marker='s', markersize=4, linewidth=1.5, alpha=0.8)

plt.title(f'Secondary Network Power Allocation Comparison\n(Moving Average, Window={window_size})', fontsize=14)
plt.xlabel('Test Sample Index (Rolling Window)', fontsize=12)
plt.ylabel('Total Secondary Power P2 (Watts)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("Result_P2_Power.png")
plt.show()
