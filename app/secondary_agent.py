from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, TypedDict, Literal, Tuple
from pydantic import BaseModel, Field 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

import math
from main import MCS, M, prompt_secondary_allocation
from state import GraphState, llm

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
          Action: Output a large negative step (e.g., -30 to -50).
        - Severity MEDIUM (-3.0 to -0.5 dB): Noticeable rate drop. 
          Action: Output a moderate negative step (e.g., -10 to -25).
        - Severity LOW (-0.5 to 0.0 dB): Just barely pushed over the cliff edge. 
          Action: Output a tiny negative step (e.g., -1 to -5).
          
        [INCREASE ACTIONS - Positive Step Values]
        - Severity LOW (2.0 to 4.0 dB): The primary is safe, and you have a small amount of excess room. 
          Action: Output a small positive step (e.g., +5 to +15).
        - Severity HIGH (Margin > 4.0 dB): The primary has a massive excess margin. You are leaving free throughput on the table. 
          Action: Output a large positive step (e.g., +20 to +50).

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
        ###################
        # Step 1
        sinr_state = []
        for i in range(M):
            sinr = (state['P2'][i] * state['direct_secondary_channels'][i]) / (1.0 + sum(state['P1']) * state['cross_primary_channels'][i])
            sinr_db = 10 * math.log10(sinr) if sinr > 0 else -999
            r = 0
            for threshold, rate in MCS:
                if sinr_db >= threshold:
                    r = rate
            sinr_state.append((sinr_db, r))
        # Step 2
        # next_sinr_target = []
        # for sinr, rate in sinr_state:
        #     for i in range(len(MCS)):
        #         if rate == MCS[i][1]:
        #             sinr_lin = 10**(MCS[i+1][0]/10.0) if i+1 < len(MCS) else 10 ** (MCS[i][0]/10.0)
        #             next_sinr_target.append(sinr_lin)

        next_sinr_target = []
        for sinr_db, rate in sinr_state:
            target_th_db = None
            
            for th, r in MCS:
                if r > rate:
                    target_th_db = th
                    break
    
            if target_th_db is None:
                target_th_db = MCS[-1][0] 
                
            sinr_lin = 10 ** (target_th_db / 10.0)
            next_sinr_target.append(sinr_lin)

        # Step 3
        p2_required = []
        for i in range(M):
            interference = sum(state['P1']) * state['cross_secondary_channels'][i]
            required_p2 = (next_sinr_target[i] * (1.0 + interference)) / state['direct_secondary_channels'][i]
            p2_required.append(required_p2)
        # Step 4
        cost = []
        for i in range(M):
            cost.append(p2_required[i] - state['P2'][i])
        # Step 5
        rank = []
        for i in range(M):
            current_rate = sinr_state[i][1]
            
            # Find the next rate (Value)
            next_rate = current_rate
            for th, r in MCS:
                if r > current_rate:
                    next_rate = r
                    break
                    
            value = next_rate - current_rate
            
            # Efficiency = Mbps gained per Watt spent. 
            # (Prevent division by zero if cost is somehow 0 or negative)
            efficiency = (value / cost[i]) if cost[i] > 0 else 0 
            
            # Store as a tuple: (efficiency, receiver_index, cost, current_rate)
            rank.append((efficiency, i, cost[i], current_rate))
            
        # Sort receivers by highest efficiency first
        rank.sort(key=lambda x: x[0], reverse=True)
        
        ###################
        # Step 6: Distribute the Budget (Knapsack)
        budget = resp.step
        new_P2 = list(state['P2'])
        
        if budget > 0:
            # INCREASE logic: Be greedy! Buy the most efficient upgrades first.
            for eff, i, cst, curr_rate in rank:
                # We need to round up the cost to ensure we actually cross the threshold
                required_watts = int(math.ceil(cst))
                
                if budget >= required_watts and required_watts > 0:
                    new_P2[i] += required_watts
                    budget -= required_watts
                    
            # If we have leftover budget that isn't enough to upgrade ANY receiver to the next level,
            # we dump it into the top-ranked receiver to get them closer for the next round.
            if budget > 0:
                top_index = rank[0][1]
                new_P2[top_index] += budget
                
        elif budget < 0:
            # DECREASE logic: The primary user is mad. We need to cut power.
            # We cut from the "excess margin" (power that isn't contributing to the current data rate).
            budget_to_cut = abs(budget)
            
            # Sort by lowest efficiency first, so we penalize the worst links
            rank.sort(key=lambda x: x[0]) 
            
            for eff, i, cst, curr_rate in rank:
                if budget_to_cut <= 0:
                    break
                
                # Find the absolute minimum power needed to maintain the CURRENT rate
                min_sinr_db = -999
                for th, r in MCS:
                    if r == curr_rate:
                        min_sinr_db = th
                        break
                        
                if min_sinr_db != -999:
                    min_sinr_lin = 10 ** (min_sinr_db / 10.0)
                    interference = sum(state['P1']) * state['cross_secondary_channels'][i]
                    min_p2 = (min_sinr_lin * (1.0 + interference)) / state['direct_secondary_channels'][i]
                    
                    # The "Donor" concept: excess power that does nothing for us
                    excess = new_P2[i] - min_p2 
                    
                    if excess > 0:
                        # Cut as much of the excess as we can without dropping our rate
                        cut = min(budget_to_cut, int(math.floor(excess)))
                        new_P2[i] -= cut
                        budget_to_cut -= cut
            
            # If we STILL need to cut power to satisfy the Primary, we are forced to drop rates.
            # We blindly subtract from whoever has power left.
            if budget_to_cut > 0:
                for i in range(M):
                    if new_P2[i] > 0:
                        cut = min(budget_to_cut, new_P2[i])
                        new_P2[i] -= cut
                        budget_to_cut -= cut

        state['P2'] = new_P2
        print(f"New power after Knapsack distribution: {state['P2']}")

    return state