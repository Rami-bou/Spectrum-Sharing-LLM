from langchain.messages import SystemMessage
from typing import List, Dict, Any, Optional, TypedDict, Literal, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import math
from environment import get_mcs_threshold, MCS, M, N, allocate_p2_knapsack_optimal, train

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

    worst_margin: float

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

prompt_secondary_allocation = build_prompt(train)

llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)

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
    2. -3.0 dB <= Margin < -0.5 dB: action=DECREASE, severity=MEDIUM.
    3. -0.5 dB <= Margin < 0.0 dB: decision=REJECT, action=DECREASE, severity=LOW.
    4. 0.0 dB <= Margin <= 0.5 dB: decision=ACCEPT.
    5. 0.5 dB < Margin <= 5.0 dB: decision=REJECT, action=INCREASE, severity=LOW.
    6. Margin > 5.0 dB: Far below capacity, secondary is being overly conservative. decision=REJECT, action=INCREASE, severity=HIGH.

    Your critique must explicitly mention the amount of the worst margin.

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
    state['worst_margin'] = worst_margin

    print(f"[Decision]: {resp.decision} ({resp.severity})")
    print(f"[Critique]: {resp.critique}")

    # if state['iteration'] > 3 and worst_margin < 0:
    #     state['P2'] = [0] * len(state['P2'])
    #     print(f"[SAFETY FALLBACK] Final allocation violates the margin ({worst_margin:.2f} dB) -- forcing P2 to zero.")
    # elif state['iteration'] > 3:
    #     print(f"[INFO] Round limit reached without ACCEPT, but margin ({worst_margin:.2f} dB) is still safe -- keeping current allocation.")
        
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
        Your goal is to adjust the total Secondary Power (P2) budget based on the Primary Evaluator's critique.
        
        The Primary Evaluator monitors the 'Worst MCS Margin' (in dB). The sweet spot is a margin exactly between 0.0 dB and 1.0 dB.
        
        You must select an integer `step` to adjust your total P2 budget strictly from the allowed lists below. 
        
        [DECREASE ACTIONS - Margin < 0.0 dB]
        Allowed Steps: [-30, -25, -20, -15, -10, -5, -3, -2, -1]
        - Worst Case Anchor (Margin <= -10.0 dB): You completely jammed the Primary. Choose the biggest step: -30.
        - Least Case Anchor (Margin = -0.1 dB): You barely crossed the threshold. Choose the smallest step: -1.
        - In Between: Evaluate where the current margin falls between -0.1 dB and -10.0 dB. If it leans closer to the worst case, pick a correspondingly larger step (e.g., -20, -25). If it leans closer to the least case, pick a smaller step (e.g., -3, -5).
        
        [INCREASE ACTIONS - Margin > 1.0 dB]
        Allowed Steps: [+1, +2, +3, +5, +10, +15, +20, +25, +30]
        - Worst Case Anchor (Margin >= +10.0 dB): The primary has massive excess margin. Choose the biggest step: +30.
        - Least Case Anchor (Margin = +1.1 dB): You are just barely above the sweet spot. Choose the smallest step: +1.
        - In Between: Evaluate where the current margin falls between +1.1 dB and +10.0 dB. If it leans toward massive excess, pick a larger step (e.g., +15, +20). If it is close to the sweet spot, pick a smaller step (e.g., +3, +5).

        CRITICAL RULES:
        1. You must ONLY select a step value from the Allowed Steps lists provided above.
        2. Always output a NEGATIVE integer if the action is DECREASE. Always output a POSITIVE integer if the action is INCREASE.
        3. Oscillation Check: Look at your `delta_hist`. If your last step caused the margin to flip polarity (e.g., from positive to negative), you jumped too far. You MUST reverse direction and pick a step from the list that is strictly smaller in magnitude than your previous step.

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
        # state['P2'] = [int(round((inv / sum_inverses) * P2_new)) for inv in inverses]
        state['P2'] = allocate_p2_knapsack_optimal(P2_new, state['direct_secondary_channels'], state['cross_secondary_channels'], state['P1'])

        print(f"New power after delta: {state['P2']}")

    return state