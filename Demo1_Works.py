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
from typing import List, Dict, Any, Optional, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
import math

f = 2e9
c = 3e8
wave = c / f
P1 = 100
N0 = -100
N0_watts = (10**(N0 / 10)) / 1000
target_SE1 = 2.0
target_val = 2 ** target_SE1
MAX_P2 = 50
number = 0
unique_samples = {}
used_hpp = set()
used_hsp = set()
used_hpp_hsp = set()

while len(unique_samples) < 1500:
    txp = [random.uniform(0, 10), random.uniform(26, 50)]
    txs = [random.uniform(0, 10), random.uniform(0, 25)]
    rxp = [random.uniform(15, 50), random.uniform(0, 25)]
    rxs = [random.uniform(15, 50), random.uniform(26, 50)]
    if rxp == rxs: continue

    dpp = np.sqrt((txp[0]-rxp[0])**2 + (txp[1]-rxp[1])**2)
    dsp = np.sqrt((txs[0]-rxp[0])**2 + (txs[1]-rxp[1])**2)
    dps = np.sqrt((txp[0]-rxs[0])**2 + (txp[1]-rxs[1])**2)
    dss = np.sqrt((txs[0]-rxs[0])**2 + (txs[1]-rxs[1])**2)

    hpp = (wave / (4 * np.pi * dpp))**2
    hsp = (wave / (4 * np.pi * dsp))**2
    hps = (wave / (4 * np.pi * dps))**2
    hss = (wave / (4 * np.pi * dss))**2

    hpp_normal = int(round(((np.log(hpp) - (-16.2)) / 1.8 * 10))) * random.randint(1, 9)
    hsp_normal = int(round(((np.log(hsp) - (-16.2)) / 1.8 * 10))) * random.randint(1, 9)
    hps_normal = int(round(((np.log(hps) - (-16.2)) / 1.8 * 10))) * random.randint(1, 9)
    hss_normal = int(round(((np.log(hss) - (-16.2)) / 1.8 * 10))) * random.randint(1, 9)
    # hps_normal = random.randint(-5, 5)
    # hss_normal = random.randint(-5, 5)

    pair = (hpp_normal, hsp_normal)
    pair1 = (hps_normal, hss_normal)

    # First two columns must be unique --> Because if they repeated with different tuples
    # it will give the same P2 value because we are using only them in the formula,
    # so the model learn that with this two first columns it will always give the same P2 value, which is not true.
    if pair in used_hpp_hsp:
        continue
    # here's not necessary, because even if it repeated, it will give different P2 values because the first two columns are different, 
    # so the model will learn a varaiant pattern, even if it's the same last two columns but the tuple is diff
    # so the P2 is different.
    if pair1 in used_hsp:
      continue
    
    A_tuple = (hpp_normal, hsp_normal, hps_normal, hss_normal)
    if A_tuple in unique_samples: 
      continue

    hpp_quant = np.exp(((hpp_normal / 10) * 1.8) - 16.2)
    hsp_quant = np.exp(((hsp_normal / 10) * 1.8) - 16.2)

    numerator = (P1 * hpp_quant) / target_val - N0_watts
    best_P2_exact = numerator / hsp_quant
    if best_P2_exact < 0:
        best_P2 = 0
    else:
        best_P2 = int(min(MAX_P2, math.floor(best_P2_exact)))

    SE1_exact = np.log2((P1 * hpp_quant) / (N0_watts + hsp_quant * best_P2))
    if 2.0 <= SE1_exact <= 10.0:
        unique_samples[A_tuple] = best_P2
        used_hpp_hsp.add(pair)
        used_hsp.add(pair1)
        number += 1
        print(A_tuple, SE1_exact, number)

normal_samples = [[k[0], k[1], k[2], k[3], v] for k, v in unique_samples.items()]
random.shuffle(normal_samples)

h_training = normal_samples[:1000]
h_testing = normal_samples[1000:]

prompt = 'You are a mathematical tool. Predict B for given A.\n'
for i in range(len(h_training)):
    prompt += f'If A is ({h_training[i][0]}, {h_training[i][1]}, {h_training[i][2]}, {h_training[i][3]}) then B is {h_training[i][4]}\n'
print(prompt)
print(len(h_training))
print(len(h_testing))
class PredictdModel(BaseModel):
    predict: int = Field(..., description="The predicted B value")

llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0)
structured_demand = llm.with_structured_output(PredictdModel)

print("##### LLM Inference #####")
count = 0
MAE = 0

plot_true_p2 = []
plot_pred_p2 = []
plot_pred_se1 = []

for sample in h_testing:
    current_channels = (sample[0], sample[1], sample[2], sample[3])
    P2_true = sample[4]

    resp = structured_demand.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Predict B for the following A: {current_channels}\nB: ")
    ])
    P2_pred = resp.predict

    hpp_val = np.exp(((sample[0] / 10) * 1.8) - 16.2)
    hsp_val = np.exp(((sample[1] / 10) * 1.8) - 16.2)

    SE1_pred = np.log2((P1 * hpp_val) / (N0_watts + hsp_val * P2_pred))
    MAE += abs(P2_pred - P2_true)
    print(f'If A is {current_channels} then B is {P2_pred} and original value {P2_true}')   
    
    if 2.0 <= SE1_pred <= 10.0:
        count += 1

    plot_true_p2.append(P2_true)
    plot_pred_p2.append(P2_pred)
    plot_pred_se1.append(SE1_pred)

print(f'Accuracy = {count/len(h_testing):.2%}')
print(f'MAE = {MAE/len(h_testing):.2f}')


true_p2_arr = np.array(plot_true_p2)
pred_p2_arr = np.array(plot_pred_p2)
pred_se1_arr = np.array(plot_pred_se1)

plt.figure(figsize=(12,6))

plt.plot(
    pred_se1_arr,
    color='royalblue',
    linewidth=1.8,
    label='Predicted SE1'
)

plt.axhline(
    y=target_SE1,
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'Threshold = {target_SE1}'
)

plt.scatter(
    np.arange(len(pred_se1_arr)),
    pred_se1_arr,
    s=12,
    color='royalblue'
)

plt.title("Primary User Spectral Efficiency Across Test Samples")
plt.xlabel("Test Sample Index")
plt.ylabel("SE1 (bps/Hz)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.savefig("SE_variation.png", dpi=300)
plt.show()