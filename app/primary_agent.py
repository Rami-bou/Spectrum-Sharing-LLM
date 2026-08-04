from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, TypedDict, Literal, Tuple
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
import math
from dataset import get_mcs_threshold
from graph import GraphState, llm

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