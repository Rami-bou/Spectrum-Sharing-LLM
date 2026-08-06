"""
This one we critique on the caused intreference on the primary receivers.
We need to draw the interference
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

# number of primary receivers
N = 4
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

repeated = set()
data = []

f = 2e9
c = 3e8
wave = c / f
P_max = 100.0

I_max = 1000
P1 = 100
scale_factor = 1e8
possible_P2 = []

primary_I_max = 1000
secondary_I_max = 1500

random.seed(10)

def get_mcs_threshold(sinr_db):
    """Finds the minimum required SINR (dB) for the current state."""
    target_th = -999
    for th, rate in MCS:
        if sinr_db >= th:
            target_th = th
        else:
            break
    return target_th

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

def calculate_primary_discrete_rate(P1_vector, P2_vector, direct_h_secondary, cross_h_secondary):
    """Calculates total Primary Throughput based on discrete MCS levels."""
    total_throughput_mbps = 0
    total_P1 = sum(P1_vector)
    
    for j in range(len(P1_vector)):
        signal = P2_vector[j] * direct_h_secondary[j]
        interference_from_secondary = total_P1 * cross_h_secondary[j]
        
        # Calculate physical linear SINR
        sinr_linear = signal / (1.0 + interference_from_secondary)
        
        # Map to discrete hardware throughput
        total_throughput_mbps += get_discrete_rate(sinr_linear)
        
    return total_throughput_mbps

def gen_channels(length):
    while len(data) < length:
        # Primary receivers: bounded cell around their own TX (10-35m),
        # kept away from the secondary transmitter's territory.
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

        # Secondary receivers: tight cluster around their own TX (1-5m).
        # Needed so secondary's own SINR can reach the same MCS table primary uses --
        # a spread-out secondary is always interference-limited to below MCS tier 0.
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

        inverses = [1.0 / v for v in direct_h_primary]
        sum_inverses = sum(inverses)
        P1_dist = [int(round((inv / sum_inverses) * allowed_p1)) for inv in inverses]

        # Reject the sample if ANY primary receiver fails to clear the lowest MCS
        # tier at baseline (P2=0) -- this is what "best P1" actually means: don't
        # silently skip weak receivers later, guarantee all of them qualify up front.
        baseline_ok = True
        for j in range(N):
            signal = P1_dist[j] * direct_h_primary[j]
            if signal <= 0 or get_mcs_threshold(10 * math.log10(signal)) < 0:
                baseline_ok = False
                break
        if not baseline_ok:
            continue

        p2_limits = []
        for j in range(N):
            signal = P1_dist[j] * direct_h_primary[j]
            baseline_sinr_db = 10 * math.log10(signal)
            target_th = get_mcs_threshold(baseline_sinr_db)  # guaranteed >= 0 now
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
        # floor (not round) guarantees sum(P2_dist) <= allowed_p2
        P2_dist = [int(math.floor((inv / sum_inverses) * allowed_p2)) for inv in inverses]
        
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

def primary(state:GraphState) -> GraphState:

    total_p2 = sum(state['P2'])

    interference_on_primary = [total_p2 * state['cross_primary_channels'][i] for i in range(len(state['cross_primary_channels']))]
    primary_gaps = [inter - primary_I_max for inter in interference_on_primary]
    max_gap = max(primary_gaps)
    print(f"Gap: {primary_gaps}")

    prompt_primary = f"""You are the Central Network Evaluator. Your absolute priority is protecting Primary users.
    You will receive the caused interference on your channel by the secondary user's power allocation.
    The Gap is defined as: Gap = caused_interference - {primary_I_max}. A positive Gap means the secondary is causing too much interference. A negative Gap means the secondary is well under the threshold and wasting power budget.
    
    Follow these exact bands based on the Gap:
    1. Gap > 1000: EMERGENCY, way too much interference. decision=REJECT, action=DECREASE, severity=HIGH.
    2. 500 <= Gap <= 999: too much interference. decision=REJECT, action=DECREASE, severity=MEDIUM.
    3. 100 < Gap <= 499: normal interference. decision=REJECT, action=DECREASE, severity=LOW.
    5. 0 < Gap <= 100: Slightly above threshold, but acceptable. decision=ACCEPT.
    4. Gap <= -500: far under the threshold, wasting a lot of power budget. decision=REJECT, action=INCREASE, severity=HIGH.
    6. -499 <= Gap <= 0: Below threshold, acceptable, but can utilize more power. decision=ACCEPT.
    7. You take the history of caused interference and you check and adapt the critique based on the valeus in there (whether they reduced near to threshold, or it increased compare with previous one).

    Your critique must explicitly restate the numeric step range for the matched band, so the secondary user knows exactly what range to work within.

    Return JSON matching the schema.
    """

    structured_critic = llm.with_structured_output(PrimaryOutput)
    resp = structured_critic.invoke([
        SystemMessage(content=prompt_primary),
        HumanMessage(content=f"""
        P2 Allocations: {state['P2']}
        Worst-Case Primary Gap: {max_gap}
        """
        # Primary Gaps (Interference - {primary_I_max}): {primary_gaps}

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
test = data[90:100]
prompt_secondary_allocation = build_prompt(train)
all_pred_P2 = []
all_true_P2 = []
se_pred_list = []
se_true_list = []

print("\nStarting Benchmark over Test Dataset...")
for i in range(len(test)):
    direct_h_sec = test[i][1] 
    cross_h_sec = test[i][3]  
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
    
    se_pred_list.append(calculate_primary_discrete_rate(true_p1, pred_p2, direct_h_sec, cross_h_sec))
    se_true_list.append(calculate_primary_discrete_rate(true_p1, true_p2, direct_h_sec, cross_h_sec))

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

pred_p2_sum = [sum(p) for p in all_pred_P2]
true_p2_sum = [sum(p) for p in all_true_P2]

plt.figure(figsize=(12, 6))
plt.plot(moving_average(true_p2_sum, window_size), label='True Optimal Secondary Power (P2)', color='blue', linestyle='--')
plt.plot(moving_average(pred_p2_sum, window_size), label='Agent Allocated Secondary Power (P2)', color='red', linestyle='-')
plt.title('Secondary Network Transmit Power Budget (P2) Comparison')
plt.xlabel('Test Sample Index')
plt.ylabel('Total Allocated P2 Power (Watts)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig("Result_P2_Power.png")

plt.show()