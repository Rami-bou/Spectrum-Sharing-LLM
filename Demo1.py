# This one they use inverse SE calc to get the exact P2 value

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
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model

txp = [10.1, 30.7]
txs = [0.0, 10.2]
f = 2e9
c = 3e8
wave = c / f
P1 = 100
N0 = -100
N0_watts = (10**(N0 / 10)) / 1000
target_SE1 = 10

target = 2 ** target_SE1 

all_good_samples = []
normal_samples = []

for i in range(1000):
    rxp = [random.uniform(15, 50), random.uniform(0, 25)]
    rxs = [random.uniform(15, 50), random.uniform(26, 50)]

    if rxp == rxs:
        rxs = [random.uniform(15, 50), random.uniform(26, 50)]

    # Distances
    dpp = np.sqrt((txp[0]-rxp[0])**2 + (txp[1]-rxp[1])**2)
    dps = np.sqrt((txp[0]-rxs[0])**2 + (txp[1]-rxs[1])**2)
    dsp = np.sqrt((txs[0]-rxp[0])**2 + (txs[1]-rxp[1])**2)
    dss = np.sqrt((txs[0]-rxs[0])**2 + (txs[1]-rxs[1])**2)

    # Channel gains
    hpp = (wave / (4 * np.pi * dpp))**2
    hps = (wave / (4 * np.pi * dps))**2
    hsp = (wave / (4 * np.pi * dsp))**2
    hss = (wave / (4 * np.pi * dss))**2

    hpp_normal = int(round(((np.log(hpp) - (-16.2)) / 1.8 * 10)))
    hsp_normal = int(round(((np.log(hsp) - (-16.2)) / 1.8 * 10)))
    hps_normal = int(round(((np.log(hps) - (-16.2)) / 1.8 * 10)))
    hss_normal = int(round(((np.log(hss) - (-16.2)) / 1.8 * 10)))

    for j in range(100):
        SE = np.log2((P1 * hpp) / (N0_watts + hsp * j))
        if SE >= target_SE1:
            best_P2 = j
        else:
            break
    
    best_P2_rounded = best_P2

    normal_samples.append([hpp_normal, hsp_normal, hps_normal, hss_normal, best_P2_rounded])
    all_good_samples.append([hpp, hsp, hps, hss, best_P2])

random.shuffle(normal_samples)
h_training = normal_samples[:800]
h_testing = normal_samples[800:]

prompt = 'Take a deep breath and work on this problem step-by-step. You are a mathematical tool to predict some model. Your job is to predict B for given A. The following is the dataset that you can use for the prediction.\n'
for i in range(len(h_training)):
    prompt += f'If A is {h_training[i][0], h_training[i][1], h_training[i][2], h_training[i][3]} then B is {h_training[i][4]}\n'
print(prompt)

class PredictdModel(BaseModel):
  predict: int = Field(..., description="The predicted B value")

llm = ChatOllama(model="dengcao/Qwen3-8B:Q5_K_M", temperature=0.5)
structured_demand = llm.with_structured_output(PredictdModel)

results_SE2_true = []
results_SE2_pred = []

print("##### LLM Inference #####")

# Assuming 'h_testing' contains your test samples: [hpp_norm, hsp_norm, hps_norm, hss_norm, best_P2_true]
for sample in h_testing:
    
    current_channels = (sample[0], sample[1], sample[2], sample[3])
    P2_true = sample[4]

    # 1. Move the LLM invocation INSIDE the loop so it predicts for EVERY sample
    resp = structured_demand.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Predict P2 for the following channels (DO NOT OUTPUT A ZERO): {current_channels}\nP2: ")
    ])
    
    P2_pred = resp.predict

    # 2. Reverse the custom log-normalization to recover the linear channel gains
    # Forward: A = int(round(((np.log(h) - (-16.2)) / 1.8 * 10)))
    # Inverse: h = exp(((A / 10) * 1.8) - 16.2)
    hpp = np.exp(((sample[0] / 10) * 1.8) - 16.2)
    hsp = np.exp(((sample[1] / 10) * 1.8) - 16.2)
    hps = np.exp(((sample[2] / 10) * 1.8) - 16.2)
    hss = np.exp(((sample[3] / 10) * 1.8) - 16.2)

    # 3. Calculate predicted SE1 using the exact formula from data generation
    # Critical: Used N0_watts instead of N0, and removed the mismatched "1 +" to align with generation math
    SE1_pred = np.log2((P1 * hpp) / (N0_watts + hsp * P2_pred))

    # 4. Evaluate QoS Constraint
    if SE1_pred >= target_SE1:
        print(f"Channels: {current_channels} -> Predicted P2 = {P2_pred:<6} GOOD (SE1 = {SE1_pred:.2f}) True P2 = {P2_true:<6}")

        # Calculate Secondary Spectral Efficiencies using consistent math
        # SE2_pred = np.log2((P2_pred * hss) / (N0_watts + hps * P1))
        # SE2_true = np.log2((P2_true * hss) / (N0_watts + hps * P1))

        # results_SE2_pred.append(SE2_pred)
        # results_SE2_true.append(SE2_true)

    else:
        print(f"Channels: {current_channels} -> Predicted P2 = {P2_pred:<6} BAD (SE1 = {SE1_pred:.2f} < {target_SE1})")