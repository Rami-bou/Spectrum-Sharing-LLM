from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, TypedDict, Literal, Tuple
from pydantic import BaseModel, Field 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

import math
from state import GraphState, llm
from dataset import gen_channels, MCS, M, allocate_p2_knapsack_optimal

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

        # 1. Calculate the new proposed TOTAL budget
        new_total_budget = total_p2 + resp.step
        
        # 2. Prevent the budget from dropping below zero
        new_total_budget = max(0, new_total_budget) 

        # 3. Use the exact same Global Knapsack function as the dataset!
        state['P2'] = allocate_p2_knapsack_optimal(
            new_total_budget, 
            state['direct_secondary_channels'], 
            state['cross_secondary_channels'], 
            state['P1']
        )
        
        print(f"New power after Global Knapsack distribution: {state['P2']}")

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

data = gen_channels(120)
train = data[:90]
test = data[90:100]
prompt_secondary_allocation = build_prompt(train)