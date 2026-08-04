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
from langgraph.graph import StateGraph, END
from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
import math
from dotenv import load_dotenv

#load_dotenv()

#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
#os.environ["LANGCHAIN_PROJECT"] = "Test Logging"

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

random.seed(10)

def gen_channels(length):
  while len(data) < length:
    txp = [random.uniform(0, 10), random.uniform(26, 50)]
    txs = [random.uniform(0, 10), random.uniform(0, 25)]
    rxp = [random.uniform(15, 50), random.uniform(0, 25)]
    rxs = [random.uniform(15, 50), random.uniform(26, 50)]

    dpp = np.sqrt((txp[0]-rxp[0])**2 + (txp[1]-rxp[1])**2)
    dsp = np.sqrt((txs[0]-rxp[0])**2 + (txs[1]-rxp[1])**2)
    dps = np.sqrt((txp[0]-rxs[0])**2 + (txp[1]-rxs[1])**2)
    dss = np.sqrt((txs[0]-rxs[0])**2 + (txs[1]-rxs[1])**2)

    hpp = (wave / (4 * np.pi * dpp))**2
    hsp = (wave / (4 * np.pi * dsp))**2
    hps = (wave / (4 * np.pi * dps))**2
    hss = (wave / (4 * np.pi * dss))**2

    hpp_normal = int(round(hpp * scale_factor, 2))
    hsp_normal = int(round(hsp * scale_factor, 2))
    hps_normal = int(round(hps * scale_factor, 2))
    hss_normal = int(round(hss * scale_factor, 2))

    tuple_data = (hpp_normal, hsp_normal)
    if tuple_data in repeated:
        continue
    repeated.add(tuple_data)

    allowed_p2 = I_max / hsp_normal
    best_P2 = int(round(min(allowed_p2, P_max), 2))
    se = np.log2(1+(P1*hpp_normal/(1+best_P2*hsp_normal)))
    if 2.0 <= se <= 5.0:
        data.append([hpp_normal, hsp_normal, hps_normal, hss_normal, best_P2])

  random.shuffle(data)

  return data

llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)

class GraphState(TypedDict):
    P1: int
    P2: int

    allocation_history: list[int]
    interference_history: list[int]
    hpp: int
    hsp: int
    hss: int

    caused_interference: int

    iteration: int
    max_iter: int

    primary_critique: str
    primary_decision: str
    primary_action: str
    primary_severity: str


class PrimaryOutput(BaseModel):
    reasoning: str = Field(description="You provide a brief reasoning before making any decision, expalaining why you will do this.")
    decision: Literal["ACCEPT", "REJECT"]
    action: Literal["INCREASE", "DECREASE"]
    critique: str = Field(description="You give the secondary a feedback on what action to do and the severity")
    severity: str = Field(description="You mention the severity level")


class SecondaryOutput(BaseModel):
    reasoning: str = Field(description="You provide a brief reasoning before making any decision, expalaining why you will do this.")
    delta: int = Field(description="Proposed delta.")


llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)


def primary(state: GraphState) -> GraphState:
    print("######################\Primary Node working...Iteration ", state['iteration'], "######################\\n")

    prompt = f"""You are a primary user in a wireless communication environment.
    You will receive the caused interference on your channel by the secondary user allocation, based on the received value and how it bigger or lesser then the 
    {I_max} you will decide whether to ACCEPT or REJECT.
    
    If your decision is REJECT you need to write a critique, telling the secondary user what action to do and the severity of this action.

    Rules:
    1. If the caused interference is bigger then {I_max} by a margin of > 1000, it's an emergency state, you will order the secondary to reduce the power with a
    segnificant amount (unitil 50%) -> severity: VERY HIGH.
    2. If the caused interference is bigger then {I_max} by a margin of > 200 until 900, you should tell him that he can increase/decease its power with a slight amount (he should be 
    careful so it doesn't exeed the threshold)-> severity: HIGH.
    3. If the caused interference is bigger then {I_max} by a margin of > 100 and < 200, in this case the sevirity id LOW, so it should use small steps < 15.
    4. If the If the caused interference is equal {I_max} or less (by a small amount like [1-100]) you just ACCEPT the power proposal.

    Return JSON matching the schema.
    """
    # calculate the values
    state['caused_interference'] = state['P2'] * state['hsp']
    state['interference_history'].append(state['caused_interference'])

    structured_critic = llm.with_structured_output(PrimaryOutput)
    resp = structured_critic.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"""
        Secondary Current P2 Proposal: {state['P2']}
        caused interference: {state['caused_interference']}
        """)
    ])

    state['primary_critique'] = resp.critique
    state['primary_decision'] = resp.decision
    state['primary_action'] = resp.action
    state['primary_severity'] = resp.severity
    state['iteration'] += 1

    print(f"Primary user {resp.decision} ({resp.severity}), P2 {state['P2']} caused {state['caused_interference']}")
    print(f"[Primary Reasoning]: {resp.reasoning}")
    print("###################################################################################################\n")

    return state


def secondary(state: GraphState) -> GraphState:
    print("######################Secondary Node working... Iteration", state['iteration'], "######################\n")

    prompt = f"""You are a secondary user in a wireless communication environment.
    Your job is to allocate a power for your self based on your channel gain quality and the channel between primary and your receiver.
    
    At first you start with the biggest value (The power values are between 1 and 100).
    
    You will receive a feedback from the primary user telling you what to do in case of exeeding its interference threshold by your power allocation.
    
    Rules:
    1. Based on the action you provide a postitive/negative value (INCREASE(positive), DECEASE(negative)).
    2. The severity will define the value of delta you will propose (In HIGH cases it should be with 10 steps, in the LOW one it should be 1 steps.
    3. You observe the P2 history and the corresponding caused interference as well so you can do a distinguish between a HIGH and a big interference and HIGH and
    a smaller interference, you have to conclude that we are getting closer to the threshold {I_max}, so the proposed lta should follow this.
    
    Return JSON matching the schema.
    """

    structured_critic = llm.with_structured_output(SecondaryOutput)
    resp = structured_critic.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"""
        Secondary Current P2 Proposal: {state['P2']}
        caused interference: {state['caused_interference']}

        Primary critique: {state["primary_critique"]}
        Severity: {state['primary_severity']}
        """)
    ])

    P2_new = int(max(1, min(100, state['P2'] + resp.delta)))
    state['P2'] = P2_new
    state['allocation_history'].append(P2_new)

    print(f"Secondary proposes delta {resp.delta:+d} -> new P2 {P2_new}")
    print(f"[Secondary Reasoning]: {resp.reasoning}")
    print("#####################################################################################################\n")

    return state


def finalizer(state: GraphState) -> Literal["revise", "finalize"]:
  print("Finalizer...\n")
  if state["iteration"] > state["max_iter"]:
    return "finalize"

  if state['primary_decision'] == "REJECT":
    return "revise"

  return "finalize"


workflow = StateGraph(GraphState)

workflow.add_node("Primary", primary)
workflow.add_node("Secondary", secondary)

workflow.set_entry_point("Primary")

workflow.add_conditional_edges(
    "Primary",
    finalizer,
    {
        "revise": "Secondary",
        "finalize": END,
    }
)
workflow.add_edge("Secondary", "Primary")

app = workflow.compile()

try:
    from IPython.display import Image, display
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception as e:
    print("Graph visualization skipped:", e)

if 'round' in globals():
    del round

above_max_interference = []
mean_error = 0
actual_SE = []
predicted_SE = []
pred_of_sec = []
within_10 = 0
Not_within_10 = 0
cause_interference = 0
# Ensure data is generated fresh
data.clear()
repeated.clear()
ch = gen_channels(1000)
print(ch)
ch = ch[200:250]

print("\n--- Starting LLM Negotiation ---")

for hpp, hsp, hps, hss, analytical_best_p2 in ch:
  initial_state = {
      "P1": P1,
      "P2": 100,
      "allocation_history": [100],
      "interference_history": [],
      "hpp": hpp,
      "hsp": hsp,
      "hss": hss,
      "caused_interference": 0,
      "iteration": 0,
      "max_iter": 5,
      "primary_critique": "",
      "primary_decision": "",
      "primary_action": "",
      "primary_severity": ""
  }
  result = app.invoke(initial_state)

  se_true = np.log2(1+(P1*hpp/(1+analytical_best_p2*hsp)))
  se_pred = np.log2(1+(P1*hpp/(1+result['P2']*hsp)))
  se_pred_sec = np.log2(1+(result['P2']*hss/(1+P1*hps)))
  print(se_true)
  print(se_pred)
  actual_SE.append(se_true)
  predicted_SE.append(se_pred)
  sumRates = se_pred + se_pred_sec
  pred_of_sec.append(sumRates)

  mean_error += abs(result['P2'] - analytical_best_p2)

  error = abs(result['P2'] - analytical_best_p2)
  if error <= 10:
    within_10 += 1
  else:
    Not_within_10 += 1


  inter = result['P2'] * hsp
  above_max_interference.append(inter)

  if inter > I_max:
    cause_interference += 1

  print(f"Final P2 Assigned by LLM: {result['P2']}")
  print(f"Analytical Best P2 was:   {analytical_best_p2}")
  print(f"Final Caused Interference: {inter:.2f}")

print(f'\nMean Error: {mean_error/len(ch):.2f}')
print(f"Prediction within +10 and -10: {within_10/len(ch):.2%}")
print(f"Prediction outside +10 and -10: {Not_within_10/len(ch):.2%}")
print(f"Cause interference: {cause_interference/len(ch):.2%}")

fig, ax1 = plt.subplots(1, figsize=(15, 6))
ax1.plot(actual_SE, color='red', label='Actual SE', marker='o', linestyle='None')
ax1.plot(predicted_SE, color='blue', label='Predicted SE', marker='x', linestyle='None')

ax1.set_title('Actual vs. Predicted SE')
ax1.set_xlabel('Data Point Index')
ax1.set_ylabel('SE Value')
ax1.set_ylim([1, 7])
ax1.legend()
ax1.grid(True)

plt.tight_layout()
plt.savefig("A2A_math_controller_SE_Result.png")
plt.show()

fig, ax1 = plt.subplots(1, figsize=(15, 6))
ax1.plot(pred_of_sec, color='red', label='Rate', marker='o', linestyle='None')

ax1.set_title('Sum Of Rates')
ax1.set_xlabel('Data Point Index')
ax1.set_ylabel('Values')
ax1.set_ylim([1, 7])
ax1.legend()
ax1.grid(True)
plt.tight_layout()
plt.savefig("A2A_math_controller_Sum_Rates.png")
plt.show()

fig2, ax2 = plt.subplots(1, figsize=(15, 6))
step = 10
indices = np.arange(0, len(actual_SE), step)
actual_SE_10 = [actual_SE[i] for i in indices]
predicted_SE_10 = [predicted_SE[i] for i in indices]

ax2.plot(indices, actual_SE_10, color='red', label='Actual SE (every 10 steps)', marker='o', linestyle='-')
ax2.plot(indices, predicted_SE_10, color='blue', label='Predicted SE (every 10 steps)', marker='x', linestyle='-')

ax2.set_title('Actual vs. Predicted SE (Every 10 Steps)')
ax2.set_xlabel('Data Point Index')
ax2.set_ylabel('SE Value')
ax2.set_ylim([1, 7])
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("A2A_math_controller_SE_Result_each_10.png")
plt.show()


fig, ax1 = plt.subplots(1, figsize=(15, 6))
ax1.plot(above_max_interference, color='blue', label='Interference', marker='x', linestyle='None')

ax1.set_title('How much interference')
ax1.set_xlabel('Data Point Index')
ax1.set_ylabel('Interference Values')
ax1.set_ylim([500, 1500])
ax1.legend()
ax1.grid(True)
plt.axhline(
    y=1000,
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'Threshold = {1000}'
)
plt.tight_layout()
plt.savefig("A2A_math_controller_Interference.png")
plt.show()