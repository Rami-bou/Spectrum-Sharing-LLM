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

N = 4  
M = 3 

data = []

f = 2e9
c = 3e8
wave = c / f
scale_factor = 1e8

secondary_I_max = 8000
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

def gen_channels(length):
    while len(data) < length:
        primary_transmitter = [8, 35]
        secondary_transmitter = [5, -20]

        # 1. Primary Channels
        dist__state = random.randint(1, 3)
        position_primary_receiver = []
        direct_h_primary = []
        for i in range(N):
            if dist__state == 1:
                rp = [random.uniform(10, 15), random.uniform(30, 50)]
            elif dist__state == 2:
                rp = [random.uniform(16, 25), random.uniform(30, 50)]
            else:
                rp = [random.uniform(25, 50), random.uniform(30, 50)]

            position_primary_receiver.append(rp)
            d = np.sqrt((rp[0]-primary_transmitter[0])**2 + (rp[1]-primary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            direct_h_primary.append(int(round(h * scale_factor, 2)))

        # 2. Secondary Channels
        dist__state = random.randint(1, 3)
        position_secondary_receiver = []
        direct_h_secondary = []
        for i in range(M):
            if dist__state == 1:
                rs = [random.uniform(10, 15), random.uniform(0, 29)]
            elif dist__state == 2:
                rs = [random.uniform(16, 25), random.uniform(0, 29)]
            else:
                rs = [random.uniform(25, 50), random.uniform(0, 29)]

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
            
            # Baseline SINR in dB (when P2 = 0)
            baseline_sinr_db = 10 * math.log10(signal)
            
            # Target MCS cliff threshold
            target_th = get_mcs_threshold(baseline_sinr_db)
            if target_th < 0:  # Skip if user can't even reach MCS 0
                continue
            
            # Max allowed linear interference before dropping below target_th
            min_linear_sinr = 10 ** (target_th / 10.0)
            max_interference = (signal / min_linear_sinr) - 1.0
            
            if max_interference > 0 and cross_h_primary[j] > 0:
                p2_limits.append(max_interference / cross_h_primary[j])

        if not p2_limits:
            continue

        allowed_p2 = int(math.floor(min(p2_limits)))
        if allowed_p2 < M:
            continue

        # 6. P2 Power Distribution
        inverses_sec = [1.0 / v for v in direct_h_secondary]
        sum_inverses_sec = sum(inverses_sec)
        P2_dist = [int(round((inv / sum_inverses_sec) * allowed_p2)) for inv in inverses_sec]

        data.append([direct_h_primary, direct_h_secondary, cross_h_primary, cross_h_secondary, P1_dist, P2_dist])

    return data

llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)

"""We start with the beamfor version, where each receivers i share the sub-channel"""
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

    # 3. LLM Prompt based on MCS Margin Bands
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
    """The primary transmitter, have more prevelige."""
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
        prompt = f"""You are a secondary user in a wireless communication environment.
        Based on the received critique, you adjust your P2 proposal.
        You add or substract depends on the action received from primary user.
        You decide the step based on the P2 history and the corresponding caused interference, and you related them with the severity, so you can know whether we are far or near to the best P2.
        The sing of the step (+ or -) depends on the action received as well.
        
        Severity-to-step-size guide (same bands the primary user uses):
        - HIGH: step magnitude roughly 20 to 30
        - MEDIUM: step magnitude roughly 10 to 20
        - LOW: step magnitude roughly 1 to 10

        Do not repeat the exact same step as your last one if the situation (gap/severity) has changed - check your own step history below.

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

  if state['primary_decision'] == "REJECT":
    return "revise"

  return "finalize"

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
test = data[90:]
prompt_secondary_allocation = build_prompt(train)
all_pred_P2 = []
all_true_P2 = []
se_pred_list = []
se_true_list = []

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
            break # Stop at the highest satisfied threshold
            
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

window_size = 5 if len(test) < 50 else 10

def moving_average(data, w):
    """Calculates the moving average shifting by 1 step at a time."""
    return np.convolve(data, np.ones(w), 'valid') / w

smoothed_se_pred = moving_average(se_pred_list, window_size)
smoothed_se_true = moving_average(se_true_list, window_size)

plt.figure(figsize=(12, 6))
plt.plot(smoothed_se_true, label=f'True Optimal Primary SE', color='blue', linestyle='--', marker='o', markersize=4)
plt.plot(smoothed_se_pred, label=f'Agent-Protected Primary SE', color='red', linestyle='-', marker='s', markersize=4)

plt.title(f'Primary Network Spectral Efficiency Comparison\n(Moving Average, Window={window_size})', fontsize=14)
plt.xlabel('Test Sample Index (Rolling Window)', fontsize=12)
plt.ylabel('Sum Spectral Efficiency (bps/Hz)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("Result_MCS.png")
plt.show()