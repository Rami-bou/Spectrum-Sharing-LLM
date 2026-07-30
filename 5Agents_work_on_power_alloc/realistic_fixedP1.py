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
    print(f"Primary Gap {primary_gaps}")

    prompt_primary = f"""You are the Central Network Evaluator. Your absolute priority is protecting Primary users.
    You will receive the caused interference on your channel by the secondary user's power allocation.
    The Gap is defined as: Gap = caused_interference - {primary_I_max}. A positive Gap means the secondary is causing too much interference. A negative Gap means the secondary is well under the threshold and wasting power budget.
    
    Follow these exact bands based on the Gap:
    1. Gap > 1050: EMERGENCY, way too much interference. decision=REJECT, action=DECREASE, severity=HIGH.
    2. 500 <= Gap <= 1050: too much interference. decision=REJECT, action=DECREASE, severity=MEDIUM.
    3. 10 <= Gap <= 499: normal interference. decision=REJECT, action=DECREASE, severity=LOW.
    4. Gap <= -500: far under the threshold, wasting a lot of power budget. decision=REJECT, action=INCREASE, severity=HIGH.
    5. 0 < Gap <= 9: Slightly above threshold, but acceptable. decision=ACCEPT.
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
        Primary Gaps (Interference - {primary_I_max}): {primary_gaps}
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

        P2_new = int(max(1, min(100, total_p2 + resp.step)))
        inverses = [1.0 / v for v in state['direct_primary_channels']]
        sum_inverses = sum(inverses)
        state['P2'] = [int(round((inv / sum_inverses) * P2_new)) for inv in inverses]

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
workflow.add_edge("Primary", END)
app = workflow.compile()

data = gen_channels(100)
train = data[:80]
test = data[80:]
prompt_secondary_allocation = build_prompt(train)
print(prompt_secondary_allocation)
for i in range(1):
    initial_state = {
        "direct_primary_channels":test[i][0], # --> array [50, 10, 4]
        "direct_secondary_channels":test[i][1],
        "cross_primary_channels":test[i][2],
        "cross_secondary_channels":test[i][3],
        "P1": [0, 0, 0, 0],
        "P2": [0, 0, 0, 0],
        "primary_critique": "",
        "secondary_critique": "",
        "primary_decision": "",
        "primary_severity": "",
        "delta_hist": [],
        "iteration": 0
    }

    result = app.invoke(initial_state)
    
    # print(f"Allocation P1 pred: {result['P1']}")
    # print(f"Allocation P1 true: {test[i][4]}")
    print(f"Deltas: {result['delta_hist']}")
    print(f"Allocation P2 pred: {result['P2']}")
    print(f"Allocation P2 true: {test[i][5]}")