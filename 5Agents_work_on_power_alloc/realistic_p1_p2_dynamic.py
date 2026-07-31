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
    """The Primary transmitter has higher privilege and evaluates secondary harm."""

    if not state.get('P1') or sum(state['P1']) == 0:
        structured_critic = llm.with_structured_output(SecondaryOutput)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt_primary_allocation),
            HumanMessage(content=f"""Complete the following allocations based on the channels:
            If the primary channels are {state['direct_primary_channels']}
            Then the Power (P1) allocation are:
            """)
        ])
        state['P1'] = resp.allocation
        print(f"[Primary] Initial P1 Set: {state['P1']}")

    else:
        total_p2 = sum(state['P2'])
        interference_on_primary = [total_p2 * h for h in state['cross_primary_channels']]
        gap = [inter - primary_I_max for inter in interference_on_primary]
        max_gap = max([inter - primary_I_max for inter in interference_on_primary])
        print(f"Gap Primary: {gap}")
        prompt_critique = f"""You are the Primary User with high privilege. 
        The current worst-case interference gap caused by the secondary user is {max_gap:.1f}.
        (Positive gap = secondary is harming you. Negative gap = you are safe, secondary has room).

        Talk to the secondary user like a human:
        - If Gap > 500: Tell them 'Hey, reduce your power, you are harming my channels!' (decision=EMERGENCY, action=DECREASE).
        - If 0 <= Gap < 500: Tell them 'Hey, reduce your power, you are harming my channels!' (decision=REJECT, action=DECREASE).
        - If Gap <= -200: Tell them 'You are well below threshold, you can increase a bit.' (decision=ACCEPT, action=INCREASE).
        - Otherwise: 'We are in a good compromise.' (decision=ACCEPT, action=KEEP).
        """

        structured_critic = llm.with_structured_output(PrimaryResponse)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt_critique),
            HumanMessage(content=f"Current P2 total: {total_p2}, Max Gap: {max_gap}")
        ])

        state['primary_critique'] = resp.critique
        state['primary_decision'] = resp.decision
        state['primary_action'] = resp.action
        
        current_p1_total = sum(state['P1'])
        new_p1_total = int(max(10, current_p1_total + resp.p1_step))
        inverses = [1.0 / v for v in state['direct_primary_channels']]
        sum_inv = sum(inverses)
        state['P1'] = [int(round((inv / sum_inv) * new_p1_total)) for inv in inverses]
        state['iteration'] += 1
        print(f"\n[Primary Talk]: {resp.critique}")
        print(f"[Primary Decision]: {resp.decision} | Action requested: {resp.action}")

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

        prompt_listener = f"""You are the Secondary Transmitter. 
        Listen to the Primary user's feedback and adjust your P2 power.

        Primary Feedback: "{state['primary_critique']}"
        Primary Decision: {state['primary_decision']}

        Decide your `p2_step`:
        - If Primary told you to reduce/harming them: output a negative integer (e.g., -15 to -30).
        - If Primary said you can increase: output a positive integer (e.g., +5 to +15).
        - If in compromise: output 0.
        """

        structured_critic = llm.with_structured_output(SecondaryResponse)
        resp = structured_critic.invoke([
            SystemMessage(content=prompt_listener),
            HumanMessage(content=f"Current P2: {state['P2']}")
        ])

        total_p2 = sum(state['P2'])
        new_p2_total = int(max(1, total_p2 + resp.p2_step))

        inverses = [1.0 / v for v in state['direct_secondary_channels']]
        sum_inverses = sum(inverses)
        state['P2'] = [int(round((inv / sum_inverses) * new_p2_total)) for inv in inverses]

        state['secondary_critique'] = resp.reasoning
        # state['secondary_critique'] = resp.critique
        print(f"[Secondary Response]: Step chosen: {resp.p2_step} | New P2: {state['P2']}")
        # print(f"[Secondary Critique]: {resp.critique}")

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
    print("Finalizer...\n")

    if sum(state['P2']) == 0:
        return "revise"

    if state['iteration'] > 4:
        return "finalize"

    if state['primary_action'] != "KEEP":
        return "revise"

    return "finalize"

workflow = StateGraph(GraphState)

workflow.add_node("Primary", primary)
workflow.add_node("Secondary", secondary)

workflow.set_entry_point("Primary")
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
test = data[81:]
prompt_primary_allocation, prompt_secondary_allocation = build_prompt(train)
print(prompt_secondary_allocation)
for i in range(1):
    initial_state = {
        "direct_primary_channels": test[i][0],
        "direct_secondary_channels": test[i][1],
        "cross_primary_channels": test[i][2],
        "cross_secondary_channels": test[i][3],
        "P1": [0] * N,
        "P2": [0] * M,
        "primary_critique": "",
        "secondary_critique": "",
        "primary_decision": "",
        "primary_action": "",
        "iteration": 0
    }

    result = app.invoke(initial_state)

    print(f"Allocation P1 pred: {result['P1']}")
    print(f"Allocation P1 true: {test[i][4]}")
    print(f"Allocation P2 pred: {result['P2']}")
    print(f"Allocation P2 true: {test[i][5]}")
    print(f"Iteration: {result['iteration']}")
