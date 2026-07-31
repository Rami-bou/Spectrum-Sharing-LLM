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
N = 3
# number of secondary receivers
M = 2

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

def gen_channels(length):
    while len(data) < length:
        # fix positions for transmitters
        primary_transmitter = [8, 35]
        secondary_transmitter = [5, 13]

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
            h_normal = int(round(h * scale_factor, 2))

            direct_h_primary.append(h_normal)

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
            h_normal = int(round(h * scale_factor, 2))

            direct_h_secondary.append(h_normal)

        cross_h_primary = []
        for pos in position_primary_receiver:
            d = np.sqrt((pos[0]-secondary_transmitter[0])**2 + (pos[1]-secondary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            h_normal = int(round(h * scale_factor, 2))
            cross_h_primary.append(h_normal)

        cross_h_secondary = []
        for pos in position_secondary_receiver:
            d = np.sqrt((pos[0]-primary_transmitter[0])**2 + (pos[1]-primary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            h_normal = int(round(h * scale_factor, 2))
            # going from primary transmitter to secondary users
            cross_h_secondary.append(h_normal)

        allowed_p1 = int(round(secondary_I_max / max(cross_h_secondary)))
        allowed_p2 = int(round(primary_I_max / max(cross_h_primary)))
        if allowed_p1 < N or allowed_p2 < M:
            continue

        # distribute the P1 accross the users where the nearest get less power and vise versa
        inverses = [1.0 / v for v in direct_h_primary]
        sum_inverses = sum(inverses)
        P1_dist = [int(round((inv / sum_inverses) * allowed_p1)) for inv in inverses]

        inverses = [1.0 / v for v in direct_h_secondary]
        sum_inverses = sum(inverses)
        P2_dist = [int(round((inv / sum_inverses) * allowed_p2)) for inv in inverses]

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
    primary_action: str

    iteration: int

class PrimaryOutput(BaseModel):
    highest_primary_gap: float = Field(description="Extract the largest number from the Primary Gaps array (remember: -50 is larger than -800).")
    highest_secondary_gap: float = Field(description="Extract the largest number from the Secondary Gaps array.")
    step_by_step_logic: str = Field(description="Write out the exact mathematical comparison for the rules using the extracted gaps before making a decision.")
    decision: Literal["ACCEPT", "REJECT"] = Field(description="The final decision based strictly on the rules.")
    target: Literal["P1", "P2", "BOTH", "NONE"] = Field(description="Which power array needs adjustment.")
    action: Literal["INCREASE", "DECREASE", "HOLD"] = Field(description="Should the target power be increased or decreased?")
    critique: str = Field(description="Explicit instructions detailing what to do with the target array.")

class SecondaryOutput(BaseModel):
    reasoning: str = Field(description="You provide a brief reasoning before making any decision, expalaining why you will do this.")
    allocation: List[int] = Field(description="Your power allocation.")

class PrimaryResponse(BaseModel):
    reasoning: str = Field(description="Your thoughts as the Primary user looking at the secondary's interference.")
    decision: Literal["ACCEPT", "REJECT"] = Field(description="Do you accept the current interference level?")
    action: Literal["INCREASE", "DECREASE", "KEEP"] = Field(description="What should the secondary do?")
    critique: str = Field(description="The message you tell the secondary user (e.g., 'Hey, reduce your power, you're hurting my channel!').")
    p1_step: int = Field(description="Adjustment step for your own P1 power if you want to optimize.")

class SecondaryResponse(BaseModel):
    reasoning: str = Field(description="Your thoughts as the Secondary user after hearing the primary's feedback.")
    p2_step: int = Field(description="The step size to add or subtract from your total P2 power.")
    critique: str = Field(description="The message you tell the primary user.")

def primary(state: GraphState) -> GraphState:
    """The Primary transmitter evaluates secondary harm and monitors its own Spectral Efficiency."""

    # --- Round 1: Initial Allocation ---
    if not state.get('P1') or sum(state['P1']) == 0:
        structured_critic = llm.with_structured_output(SecondaryOutput)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt_primary_allocation),
            HumanMessage(content=f"If the primary channels are {state['direct_primary_channels']}...")
        ])
        state['P1'] = resp.allocation
        print(f"[Primary] Initial P1 Set: {state['P1']}")
        return state  # Exit early on round 1 so state flows cleanly

    # --- Round 2+: Evaluation & Spectral Efficiency check ---
    else:
        # 1. Calculate Spectral Efficiency accurately
        sinrs = [
            (state['P1'][i] * state['direct_primary_channels'][i]) / 
            (sum(state['P2']) * state['cross_primary_channels'][i] + 1e-6)
            for i in range(len(state['direct_primary_channels']))
        ]
        se = float(np.sum([np.log2(1 + s) for s in sinrs]))
        accept = (2.0 <= se <= 5.0)

        # 2. Compute Interference Gaps
        total_p2 = sum(state['P2'])
        interference_on_primary = [total_p2 * h for h in state['cross_primary_channels']]
        gap = [inter - primary_I_max for inter in interference_on_primary]
        max_gap = max(gap)
        print(f"Gap Primary: {gap} | Current SE: {se:.2f} (Acceptable: {accept})")

        prompt_critique = f"""You are the Primary User with high privilege. 
        Current Interference Max Gap: {max_gap:.1f}
        Current Spectral Efficiency (SE): {se:.2f} (Target Range: 2.0 to 5.0)

        Rules for `primary_decision` and `action`:
        - Gap > 500: (decision=EMERGENCY, action=DECREASE).
        - 0 <= Gap <= 500: (decision=REJECT, action=DECREASE).
        - Gap <= -200: (decision=ACCEPT, action=INCREASE).
        - Otherwise: (decision=ACCEPT, action=KEEP).

        Rules for your own `p1_step` power adjustment:
        - If SE < 2.0: output positive integer (e.g., +10 to +20) to boost your signal.
        - If SE > 5.0: output negative integer (e.g., -10 to -20) to save power.
        - If 2.0 <= SE <= 5.0: output 0.
        """

        structured_critic = llm.with_structured_output(PrimaryResponse)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt_critique),
            HumanMessage(content=f"Current P2 total: {total_p2}, Secondary message: '{state.get('secondary_critique', 'None')}'")
        ])

        state['primary_critique'] = resp.critique
        state['primary_decision'] = resp.decision
        state['primary_action'] = resp.action
        
        # Adjust P1 based on SE requirements
        current_p1_total = sum(state['P1'])
        new_p1_total = int(max(10, current_p1_total + resp.p1_step))
        inverses = [1.0 / v for v in state['direct_primary_channels']]
        sum_inv = sum(inverses)
        state['P1'] = [int(round((inv / sum_inv) * new_p1_total)) for inv in inverses]
        
        state['iteration'] += 1
        print(f"\n[Primary Talk]: {resp.critique}")
        print(f"[Primary Decision]: {resp.decision} | Action requested: {resp.action}")
        print(f"[Primary Step P1]: {resp.p1_step} | New P1: {state['P1']}")

    return state

def secondary(state: GraphState) -> GraphState:
    """The Secondary transmitter negotiates, listens to primary complaints, and yields if deadlocked."""

    if not state.get('P2') or sum(state['P2']) == 0:
        structured_critic = llm.with_structured_output(SecondaryOutput)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt_secondary_allocation),
            HumanMessage(content=f"""Complete the following allocations based on the channels:
            If the secondary channels are {state['direct_secondary_channels']}
            Then the Power (P2) allocation are:  
            """)
        ])
        state['P2'] = resp.allocation
        print(f"[Secondary] Initial P2 Set: {state['P2']}")

    else:
        if state['primary_decision'] == "EMERGENCY":
            print("\n[Arbitration]: Negotiation stuck in deadlock! Secondary makes the ultimate sacrifice.")
            state['P2'] = [1 for _ in state['P2']]
            state['secondary_critique'] = "I am sacrificing my power to yield to the primary user."
        
        total_p1 = sum(state['P1'])
        interference_on_primary = [total_p1 * h for h in state['cross_secondary_channels']]
        gap = [inter - secondary_I_max for inter in interference_on_primary]
        max_gap = max([inter - primary_I_max for inter in interference_on_primary])

        prompt_listener = f"""You are the Secondary Transmitter. 
        Listen to the Primary user's feedback and adjust your P2 power.

        Primary Feedback: "{state['primary_critique']}"
        Primary Decision: {state['primary_decision']}

        Decide your `p2_step`:
        - If Primary told you to reduce/harming them: output a negative integer (e.g., -15 to -30).
        - If Primary said you can increase: output a positive integer (e.g., +5 to +15).
        - If in compromise: output 0.

        You can provide a guidance message to the primary if your Gap > 500, so he can be gentle (His step should be from -30 to -50).
        """

        structured_critic = llm.with_structured_output(SecondaryResponse)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt_listener),
            HumanMessage(content=f"""Current P2: {state['P2']}
            Max Gap: {max_gap}
            """)
        ])

        total_p2 = sum(state['P2'])
        new_p2_total = int(max(1, total_p2 + resp.p2_step))

        inverses = [1.0 / v for v in state['direct_secondary_channels']]
        sum_inverses = sum(inverses)
        state['P2'] = [int(round((inv / sum_inverses) * new_p2_total)) for inv in inverses]

        state['secondary_critique'] = resp.reasoning
        state['secondary_critique'] = resp.critique
        print(f"[Secondary Response]: Step chosen: {resp.p2_step} | New P2: {state['P2']}")
        print(f"[Secondary Step]: {resp.p2_step}")
        print(f"[Secondary Critique]: {resp.critique}")

    return state

def build_prompt(train):
    prompt_primary = f"""You are the primary transmitter in a wireless communication scenario.
    Your job is to allocate a transmission power for each one of your receivers.
    Here is some examples on good allocations based on the channel states:\n
    """
    for i in range(len(train)):
        prompt_primary += f"""
        If the primary channels are {train[i][0]}
        Then the Power (P1) allocation are: {train[i][4]}   
        """

    prompt_secondary = f"""You are the secondary transmitter in a wireless communication scenario.
    Your job is to allocate a transmission power for each one of your receivers.
    Here is some examples on good allocations based on the channel states:\n
    """
    for i in range(len(train)):
        prompt_secondary += f"""
        If the secondary channels are {train[i][1]}
        Then the Power (P2) allocation are: {train[i][5]} 
        """

    prompt_primary += "\nReturn JSON matching the schema."
    prompt_secondary += "\nReturn JSON matching the schema."

    return prompt_primary, prompt_secondary

def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
    print("Finalizer checking state...\n")

    if state['primary_decision'] in ["REJECT", "EMERGENCY", ""]:
        if state['iteration'] <= 2:
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

data = gen_channels(100)
train = data[:80]
test = data[80:]
prompt_primary_allocation, prompt_secondary_allocation = build_prompt(train)
se1_pred_list, se1_true_list = [], []
se2_pred_list, se2_true_list = [], []
int1_pred_list, int1_true_list = [], []
int2_pred_list, int2_true_list = [], []

num_tests = len(test)  
print(f"Starting evaluation over {num_tests} test cases...")

for i in range(num_tests):
    print(f"\n--- Evaluating Test Sample {i+1}/{num_tests} ---")
    
    dir_h_p = test[i][0]
    dir_h_s = test[i][1]
    cross_h_p = test[i][2]
    cross_h_s = test[i][3]
    p1_true = test[i][4]
    p2_true = test[i][5]
    
    initial_state = {
        "direct_primary_channels": dir_h_p,
        "direct_secondary_channels": dir_h_s,
        "cross_primary_channels": cross_h_p,
        "cross_secondary_channels": cross_h_s,
        "P1": p1_true, 
        "P2": [0] * M,
        "primary_critique": "",
        "secondary_critique": "",
        "primary_decision": "",
        "primary_action": "",
        "iteration": 0
    }

    # Run the graph
    result = app.invoke(initial_state)
    p1_pred = result['P1']
    p2_pred = result['P2']

    def calc_se(P_target, dir_h, P_interferer, cross_h):
        total_interferer = sum(P_interferer)
        sinrs = [
            (P_target[j] * dir_h[j]) / (total_interferer * cross_h[j] + 1e-6)
            for j in range(len(dir_h))
        ]
        return float(np.sum([np.log2(1 + s) for s in sinrs]))

    se1_pred_list.append(calc_se(p1_pred, dir_h_p, p2_pred, cross_h_p))
    se1_true_list.append(calc_se(p1_true, dir_h_p, p2_true, cross_h_p))
    
    se2_pred_list.append(calc_se(p2_pred, dir_h_s, p1_pred, cross_h_s))
    se2_true_list.append(calc_se(p2_true, dir_h_s, p1_true, cross_h_s))

    int1_pred_list.append(max([sum(p2_pred) * h for h in cross_h_p]))
    int1_true_list.append(max([sum(p2_true) * h for h in cross_h_p]))
    
    int2_pred_list.append(max([sum(p1_pred) * h for h in cross_h_s]))
    int2_true_list.append(max([sum(p1_true) * h for h in cross_h_s]))

def moving_average(data, window_size=5):
    if len(data) < window_size:
        return data 
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window = min(5, num_tests) 

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'CRN Power Allocation Performance (Moving Average Window = {window})', fontsize=16)

# Figure 1: SE Primary
axs[0, 0].plot(moving_average(se1_pred_list, window), label='Predicted', marker='o', alpha=0.7)
axs[0, 0].plot(moving_average(se1_true_list, window), label='True', marker='x', alpha=0.7)
axs[0, 0].set_title('Spectral Efficiency: Primary (P1)')
axs[0, 0].set_ylabel('SE (bits/s/Hz)')
axs[0, 0].set_xlabel('Test Sample (Windowed)')
axs[0, 0].legend()
axs[0, 0].grid(True, linestyle='--', alpha=0.6)

axs[0, 1].plot(moving_average(se2_pred_list, window), label='Predicted', marker='o', alpha=0.7)
axs[0, 1].plot(moving_average(se2_true_list, window), label='True', marker='x', alpha=0.7)
axs[0, 1].set_title('Spectral Efficiency: Secondary (P2)')
axs[0, 1].set_ylabel('SE (bits/s/Hz)')
axs[0, 1].set_xlabel('Test Sample (Windowed)')
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle='--', alpha=0.6)

axs[1, 0].plot(moving_average(int1_pred_list, window), label='Predicted Interference', alpha=0.8)
axs[1, 0].plot(moving_average(int1_true_list, window), label='True Interference', alpha=0.8)
axs[1, 0].axhline(y=primary_I_max, color='r', linestyle='-', linewidth=2, label=f'Threshold ({primary_I_max})')
axs[1, 0].set_title('Worst-Case Interference on Primary Receivers')
axs[1, 0].set_ylabel('Interference Level')
axs[1, 0].set_xlabel('Test Sample (Windowed)')
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle='--', alpha=0.6)

axs[1, 1].plot(moving_average(int2_pred_list, window), label='Predicted Interference', alpha=0.8)
axs[1, 1].plot(moving_average(int2_true_list, window), label='True Interference', alpha=0.8)
axs[1, 1].axhline(y=secondary_I_max, color='r', linestyle='-', linewidth=2, label=f'Threshold ({secondary_I_max})')
axs[1, 1].set_title('Worst-Case Interference on Secondary Receivers')
axs[1, 1].set_ylabel('Interference Level')
axs[1, 1].set_xlabel('Test Sample (Windowed)')
axs[1, 1].legend()
axs[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

file_name = "crn_performance_results.png"
plt.savefig(file_name, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully as {file_name}")

plt.show()