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
        
        print("Allowed P1", allowed_p1)
        print("Allowed P2", allowed_p2)
        # distribute the P1 accross the users where the nearest get less power and vise versa
        inverses = [1.0 / v for v in direct_h_primary]
        sum_inverses = sum(inverses)
        P1_dist = [int(round((inv / sum_inverses) * allowed_p1)) for inv in inverses]

        inverses = [1.0 / v for v in direct_h_secondary]
        sum_inverses = sum(inverses)
        P2_dist = [int(round((inv / sum_inverses) * allowed_p2)) for inv in inverses]

        data.append([direct_h_primary, direct_h_secondary, cross_h_primary, cross_h_secondary, P1_dist, P2_dist])

    
    return data


# d1 = gen_channels(2)
# for i in range(len(d1)):
    # direct_primary_channels = d1[i][0]
    # direct_secondary_channels = d1[i][1]
    # cross_primary_channels = d1[i][2]
    # cross_secondary_channels = d1[i][3]
    # P1_distribution = d[i][3]
    # P2_distribution = d[i][4]

llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)

"""We start with the beamfor version, where each receivers i share the sub-channel"""
class GraphState(TypedDict):
    direct_primary_channels: List[int]
    direct_secondary_channels: List[int]
    cross_primary_channels: List[int]
    cross_secondary_channels: List[int]

    P1: List[int]
    P2: List[int]

    primary_gaps: List[float]
    secondary_gaps: List[float]

    primary_critique: str
    secondary_critique: str

    primary_decision: str
    primary_severity: str

def primary(state:GraphState) -> GraphState:
    # interference caused by secondary on primary receivers (subchannel)
    P2 = 0
    P2 += (state['P2'][i] for i in range(len(state['P2'])))
    caused_interference = [P2 * state['cross_primary_channels'][i] for i in range(len(state['cross_primary_channels']))]
    primary_gaps = [inter - primary_I_max for inter in caused_interference]
    state['primary_gaps'] = primary_gaps
    print(caused_interference)
    print(primary_gaps)

    # prompt_primary = f"""ou are the Primary Network Evaluator protecting primary users from interference.
    # The maximum allowed interference threshold per receiver is {primary_I_max}.
    # Rules:
    # If Gap > 1000
    # """
    
    return state

class SecondaryOutput(BaseModel):
    reasoning: str = Field(description="You provide a brief reasoning before making any decision, expalaining why you will do this.")
    allocation_primary: List[int] = Field(description="Your allocation for all the primary receivers.")
    allocation_secondary: List[int] = Field(description="Your allocation for all of your secondary receivers.")

def secondary(state:GraphState) -> GraphState:
    """The primary transmitter, have more prevelige."""
    if not state['primary_critique']:
        structured_critic = llm.with_structured_output(SecondaryOutput)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt_secondary_allocation),
            HumanMessage(content=f"""
            If the primary channels are h_pp: {state['direct_primary_channels']}
            If the primary cross channels are h_ps: {state['cross_primary_channels']}
            The the Power (P1) allocation are:

            If your channels are h_ss: {state['direct_secondary_channels']}
            If your cross channels are h_sp: {state['cross_secondary_channels']}
            The the Power (P2) allocation are:  
            """
            )
        ])

        state['P1'] = resp.allocation_primary
        state['P2'] = resp.allocation_secondary

    return state

def build_prompt(train):
    prompt_primary = f"""You are the secondary transmitter in a wireless communication scenario.
    Your job is to allocate a transmission power for each one of your receivers and the primary receivers as well.
    Here is some examples on good allocations based on the channel states:\n
    """
    for i in range(len(train)):
        prompt_primary += f"""
        If the primary channels are h_pp: {train[i][0]}
        If the primary cross channels are h_ps: {train[i][2]}
        The the Power (P1) allocation are: {train[i][4]}

        If your channels are h_ss: {train[i][1]}
        If your cross channels are h_sp: {train[i][3]}
        The the Power (P2) allocation are: {train[i][5]}    
        """
    
    prompt_primary += "\nReturn JSON matching the schema."

    return prompt_primary

workflow = StateGraph(GraphState)
workflow.add_node("Primary", primary)
workflow.add_node("Secondary", secondary)
workflow.set_entry_point("Secondary")
workflow.add_edge("Secondary", "Primary")
workflow.add_edge("Primary", END)
app = workflow.compile()

data = gen_channels(100)
train = data[:80]
test = data[80:]
prompt_secondary_allocation = build_prompt(train)

for i in range(1):
    initial_state = {
        "direct_primary_channels":test[i][0], # --> array [50, 10, 4]
        "direct_secondary_channels":test[i][1],
        "cross_primary_channels":test[i][2],
        "cross_secondary_channels":test[i][3],
        "P1": [0, 0, 0, 0],
        "P2": [0, 0, 0, 0],
        "primary_gaps": [],
        "secondary_gaps": [],
        "primary_critique": "",
        "secondary_critique": "",
        "primary_decision": "",
        "primary_severity": ""
    }

    result = app.invoke(initial_state)
    
    print(f"Allocation P1 pred: {result['P1']}")
    print(f"Allocation P1 true: {test[i][4]}")
    print(f"Allocation P2 pred: {result['P2']}")
    print(f"Allocation P2 true: {test[i][5]}")
