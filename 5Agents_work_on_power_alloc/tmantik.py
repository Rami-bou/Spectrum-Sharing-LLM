import matplotlib.pyplot as plt
import random
import numpy as np
import math
from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

# --- 1. SYSTEM CONSTANTS ---
N = 3
M = 3
f = 2e9
c = 3e8
wave = c / f
scale_factor = 1e8

secondary_I_max = 3000
primary_I_max = 1000
random.seed(11)

MCS = [
    (2.0, 15), (5.0, 30), (9.0, 45), (11.0, 60), 
    (15.0, 90), (18.0, 120), (20.0, 150)
]

def get_mcs_threshold(sinr_db):
    target_th = -999
    for th, rate in MCS:
        if sinr_db >= th:
            target_th = th
        else:
            break
    return target_th

def get_discrete_rate(sinr_linear):
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

# --- 2. DATA GENERATION (PURE OPTIMAL BASELINE) ---
def gen_channels(length):
    data = []
    while len(data) < length:
        primary_transmitter = [50, 50]
        secondary_transmitter = [30, 30]

        direct_h_primary = []
        for i in range(N):
            rp = [random.uniform(10, 90), random.uniform(10, 90)]
            d = np.sqrt((rp[0]-primary_transmitter[0])**2 + (rp[1]-primary_transmitter[1])**2)
            direct_h_primary.append(int(round(((wave / (4 * np.pi * d))**2) * scale_factor, 2)))

        position_secondary_receiver = []
        direct_h_secondary = []
        for i in range(M):
            rs = [random.uniform(10, 50), random.uniform(10, 50)]
            position_secondary_receiver.append(rs)
            d = np.sqrt((rs[0]-secondary_transmitter[0])**2 + (rs[1]-secondary_transmitter[1])**2)
            direct_h_secondary.append(int(round(((wave / (4 * np.pi * d))**2) * scale_factor, 2)))

        cross_h_primary = []
        for i in range(N):
            d = np.sqrt((random.uniform(10, 90)-secondary_transmitter[0])**2 + (random.uniform(10, 90)-secondary_transmitter[1])**2)
            cross_h_primary.append(int(round(((wave / (4 * np.pi * d))**2) * scale_factor, 2)))

        cross_h_secondary = []
        for pos in position_secondary_receiver:
            d = np.sqrt((pos[0]-primary_transmitter[0])**2 + (pos[1]-primary_transmitter[1])**2)
            cross_h_secondary.append(int(round(((wave / (4 * np.pi * d))**2) * scale_factor, 2)))

        # P1 Optimal Allocation
        allowed_p1 = int(round(secondary_I_max / max(cross_h_secondary)))
        if allowed_p1 < N: continue

        inverses = [1.0 / v for v in direct_h_primary]
        P1_dist = [int(round((inv / sum(inverses)) * allowed_p1)) for inv in inverses]

        # P2 Optimal Limit Calculation (Strict Threshold)
        p2_limits = []
        for j in range(N):
            signal = P1_dist[j] * direct_h_primary[j]
            if signal <= 0: continue
            
            baseline_sinr_db = 10 * math.log10(signal)
            target_th = get_mcs_threshold(baseline_sinr_db)
            if target_th < 0: continue
            
            min_linear_sinr = 10 ** (target_th / 10.0)
            max_interference = (signal / min_linear_sinr) - 1.0
            
            if max_interference > 0 and cross_h_primary[j] > 0:
                p2_limits.append(max_interference / cross_h_primary[j])

        if not p2_limits: continue

        # 0.99 safety buffer to ensure optimal never causes accidental interference via rounding
        allowed_p2 = int(math.floor(min(p2_limits) * 0.99))
        if allowed_p2 < M: continue

        inverses = [1.0 / v for v in direct_h_secondary]
        P2_dist = [int(math.floor((inv / sum(inverses)) * allowed_p2)) for inv in inverses]
        
        data.append([direct_h_primary, direct_h_secondary, cross_h_primary, cross_h_secondary, P1_dist, P2_dist])

    return data

# --- 3. EVALUATION FUNCTION (SECONDARY RATE) ---
def calculate_secondary_discrete_rate(P1_vector, P2_vector, direct_h_secondary, cross_h_secondary):
    """Calculates total Secondary Throughput (Our Target Objective)"""
    total_throughput = 0
    total_P1 = sum(P1_vector)
    
    for i in range(len(P2_vector)):
        signal = P2_vector[i] * direct_h_secondary[i]
        interference_from_primary = total_P1 * cross_h_secondary[i]
        sinr_linear = signal / (1.0 + interference_from_primary)
        total_throughput += get_discrete_rate(sinr_linear)
        
    return total_throughput

# --- 4. LLM GRAPH DEFINITIONS ---
llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)

class GraphState(TypedDict):
    direct_primary_channels: List[int]
    direct_secondary_channels: List[int]
    cross_primary_channels: List[int]
    cross_secondary_channels: List[int]
    P1: List[int]
    P2: List[int]
    primary_critique: str
    primary_decision: str
    delta_hist: List[int]
    iteration: int

class PrimaryOutput(BaseModel):
    decision: Literal["ACCEPT", "REJECT"] = Field(description="Final decision")
    action: Literal["INCREASE", "DECREASE"] = Field(description="Direction of target power correction")
    severity: Literal["HIGH", "MEDIUM", "LOW"] = Field(description="Magnitude of correction")
    critique: str = Field(description="Explicit step range instructions")

class SecondaryOutput(BaseModel):
    reasoning: str = Field(description="Brief reasoning")
    allocation_secondary: List[int] = Field(description="P2 allocation array")

class SecondaryRemainRounds(BaseModel):
    reasoning: str = Field(description="Brief reasoning")
    step: int = Field(description="Step adjustment for total P2")

def primary(state: GraphState) -> GraphState:
    """Primary strictly acts as an Interference Monitor."""
    total_p2 = sum(state['P2'])
    margins = []
    
    for j in range(len(state['P1'])):
        signal = state['P1'][j] * state['direct_primary_channels'][j]
        if signal <= 0: continue
            
        baseline_sinr_db = 10 * math.log10(signal)
        target_th = get_mcs_threshold(baseline_sinr_db)
        if target_th < 0: continue
            
        interference = total_p2 * state['cross_primary_channels'][j]
        actual_sinr_linear = signal / (1.0 + interference)
        actual_sinr_db = 10 * math.log10(actual_sinr_linear) if actual_sinr_linear > 0 else -999
        
        margin = actual_sinr_db - target_th
        margins.append(margin)

    worst_margin = min(margins) if margins else -999.0

    prompt_primary = f"""You are the Primary Network Interference Monitor.
    Your ONLY job is to protect the Primary users' current data rates from Secondary interference.
    
    You evaluate the 'Worst MCS Margin' (dB). This is the Interference Headroom.
    - A positive Margin means the Primary user absorbed the interference safely.
    - A negative Margin means your interference pushed the Primary user off their MCS cliff, causing data loss.
    
    Follow these exact decision bands:
    1. Margin < -3.0 dB: EMERGENCY. decision=REJECT, action=DECREASE, severity=HIGH.
    2. -3.0 dB <= Margin < 0.0 dB: decision=REJECT, action=DECREASE, severity=MEDIUM.
    3. 0.0 dB <= Margin <= 1.0 dB: PERFECT. decision=ACCEPT.
    4. 1.0 dB < Margin <= 4.0 dB: decision=REJECT, action=INCREASE, severity=LOW.
    5. Margin > 4.0 dB: decision=REJECT, action=INCREASE, severity=HIGH.

    Your critique must explicitly state the numeric step range.
    """

    resp = llm.with_structured_output(PrimaryOutput).invoke([
        SystemMessage(content=prompt_primary),
        HumanMessage(content=f"P2 Proposed: {state['P2']} | Worst Margin: {worst_margin:.2f} dB")
    ])

    state['primary_critique'] = resp.critique
    state['primary_decision'] = resp.decision
    state['iteration'] += 1
    return state

def secondary(state: GraphState) -> GraphState:
    """Secondary uses full context for initial guess, then refines based on interference critique."""
    if not state['primary_critique']:
        prompt = f"""You are the Secondary Network Optimizer.
        Allocate optimal Secondary Power (P2) based on the full network state. 
        Pattern match against the provided optimal examples.
        """
        resp = llm.with_structured_output(SecondaryOutput).invoke([
            SystemMessage(content=prompt_secondary_allocation),
            HumanMessage(content=f"""
            Primary Direct: {state['direct_primary_channels']}
            Primary Cross (Interference path): {state['cross_primary_channels']}
            P1 Array: {state['P1']}
            Secondary Direct: {state['direct_secondary_channels']}
            --> Output Optimal P2 Array:
            """)
        ])
        state['P2'] = resp.allocation_secondary
    else:
        prompt = f"""You are the Secondary Network Optimizer. Adjust P2 budget based on Primary Interference Margin.
        DECREASE (Negative step): -20 to -30 for HIGH severity, -5 to -15 for MEDIUM.
        INCREASE (Positive step): +5 to +15 for LOW severity, +20 to +30 for HIGH.
        """
        resp = llm.with_structured_output(SecondaryRemainRounds).invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Current P2: {state['P2']} | History: {state['delta_hist']} | Critique: {state['primary_critique']}")
        ])
        
        state['delta_hist'].append(resp.step)
        P2_new = int(max(1, sum(state['P2']) + resp.step))
        inverses = [1.0 / v for v in state['direct_secondary_channels']]
        state['P2'] = [int(round((inv / sum(inverses)) * P2_new)) for inv in inverses]

    return state

def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
    # 3 ROUNDS STRICT LIMIT - The LLM must be smart enough to get it quickly!
    if state["iteration"] >= 3 or state['primary_decision'] == "ACCEPT":
        return "finalize"
    return "revise"

workflow = StateGraph(GraphState)
workflow.add_node("Primary", primary)
workflow.add_node("Secondary", secondary)
workflow.set_entry_point("Secondary")
workflow.add_edge("Secondary", "Primary")
workflow.add_conditional_edges("Primary", finalizer, {"revise": "Secondary", "finalize": END})
app = workflow.compile()

# --- 5. INITIALIZATION & FEW-SHOT PROMPT CREATION ---
print("Generating Data...")
data = gen_channels(120)
train = data[:90]
test = data[90:100]

def build_prompt(train):
    prompt = """You are the secondary transmitter. Predict the optimal P2 allocation based on full network constraints:\n"""
    for d in train:
        prompt += f"Pri Direct: {d[0]} | Pri Cross: {d[2]} | P1: {d[4]} | Sec Direct: {d[1]} --> Optimal P2: {d[5]}\n"
    return prompt

prompt_secondary_allocation = build_prompt(train)

# --- 6. TESTING LOOP & EVALUATION ---
sec_pred_rate_list = []
sec_true_rate_list = []
worst_margin_list = [] # Tracks if we broke the rules!

print("\nStarting Benchmark...")
for i, t in enumerate(test):
    initial_state = {
        "direct_primary_channels": t[0], "direct_secondary_channels": t[1],
        "cross_primary_channels": t[2], "cross_secondary_channels": t[3],
        "P1": t[4], "P2": [0]*M,
        "primary_critique": "", "primary_decision": "", "delta_hist": [], "iteration": 0
    }

    result = app.invoke(initial_state)
    pred_p2 = result['P2']
    true_p2 = t[5]
    
    # Track Objective (Secondary Rate)
    sec_pred_rate_list.append(calculate_secondary_discrete_rate(t[4], pred_p2, t[1], t[3]))
    sec_true_rate_list.append(calculate_secondary_discrete_rate(t[4], true_p2, t[1], t[3]))

    # Track Constraint (Did LLM P2 cause harmful interference?)
    margins = []
    for j in range(N):
        signal = t[4][j] * t[0][j]
        baseline_db = 10 * math.log10(signal) if signal > 0 else -999
        target_th = get_mcs_threshold(baseline_db)
        interference = sum(pred_p2) * t[2][j]
        actual_db = 10 * math.log10(signal / (1.0 + interference)) if signal > 0 else -999
        margins.append(actual_db - target_th)
    
    worst_margin_list.append(min(margins) if margins else 0)
    print(f"Sample {i+1} | Target P2: {true_p2} | LLM P2: {pred_p2} | Margin: {min(margins):.2f} dB")

# --- 7. PLOTTING THE GOLD STONE RESULTS ---
w = 5 if len(test) < 50 else 10
def ma(d): return np.convolve(d, np.ones(w), 'valid') / w

# Plot 1: Objective (Secondary Capacity)
plt.figure(figsize=(10, 5))
plt.plot(ma(sec_true_rate_list), label='True Optimal Secondary Rate', color='blue', linestyle='--', marker='o')
plt.plot(ma(sec_pred_rate_list), label='LLM Allocated Secondary Rate', color='red', linestyle='-', marker='s')
plt.title('Objective Evaluation: Secondary Network Throughput')
plt.ylabel('Sum Rate (Mbps)')
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig("Result_Secondary_Rate.png")

# Plot 2: Constraint (Interference Margin)
plt.figure(figsize=(10, 5))
plt.plot(worst_margin_list, label='LLM Interference Margin', color='purple', marker='x')
plt.axhline(y=0, color='red', linestyle='--', label='Interference Threshold (0 dB)')
plt.title('Constraint Check: Primary Network Interference Safety')
plt.ylabel('Worst MCS Margin (dB) - Above 0 is Safe')
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig("Result_Primary_Interference.png")
plt.show()