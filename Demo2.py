# This one gave us a 50% accuracy, let's see if we can improve it by using a different approach

from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
import random
from langchain.chat_models import init_chat_model
import numpy as np
from dotenv import load_dotenv
import os

# Fixed positions for transmitters
txp = [10.1, 43.7]
txs = [14.8, 20.2]

# Use a set for deduplication - store tuples of h values
seen_h_combinations = set()
all_good_h = []

# Generate more samples to ensure we have enough unique ones
while len(all_good_h) < 1200:
   
    rxp = [random.uniform(15, 50), random.uniform(0, 24)]
    rxs = [random.uniform(15, 50), random.uniform(25, 50)]

    # Calculate distances
    dpp = np.sqrt((txp[0] - rxp[0])**2 + (txp[1] - rxp[1])**2)
    dps = np.sqrt((txp[0] - rxs[0])**2 + (txp[1] - rxs[1])**2)
    dsp = np.sqrt((txs[0] - rxp[0])**2 + (txs[1] - rxp[1])**2)
    dss = np.sqrt((txs[0] - rxs[0])**2 + (txs[1] - rxs[1])**2)

    hpp = 1.0 / (4 * np.pi * dpp)**2
    hps = 1.0 / (4 * np.pi * dps)**2
    hsp = 1.0 / (4 * np.pi * dsp)**2
    hss = 1.0 / (4 * np.pi * dss)**2

    # Normalize values to integer range
    hpp_normal = int(round(((np.log(hpp) - (-16.2)) / 1.8 * 10)))
    hsp_normal = int(round(((np.log(hsp) - (-16.2)) / 1.8 * 10)))
    hps_normal = int(round(((np.log(hps) - (-16.2)) / 1.8 * 10)))
    hss_normal = int(round(((np.log(hss) - (-16.2)) / 1.8 * 10)))

    h = (hpp_normal, hsp_normal, hps_normal, hss_normal)

    if h in seen_h_combinations:
        continue

    valid_j_values = []
    for j in range(100):
        sinr = hpp_normal * 100 / (1.0 + hsp_normal * j)
        print(f"Checking j={j}: SINR={sinr}")
        if sinr < 60:
            valid_j_values.append(j)

    if valid_j_values:
        selected_j = random.choice(valid_j_values)

        seen_h_combinations.add(h)
        all_good_h.append([hpp_normal, hsp_normal, hps_normal, hss_normal, selected_j])

random.shuffle(all_good_h)

h_training = all_good_h[:1000]
h_testing = all_good_h[1000:1200]

prompt = 'Take a deep breath and work on this problem step-by-step. You are a mathematical tool to predict some model. Your job is to predict B for given A. The following is the dataset that you can use for the prediction.\n'
for i in range(len(h_training)):
    prompt += f'If A is {h_training[i][0], h_training[i][1], h_training[i][2], h_training[i][3]}, 50 then B is {h_training[i][4]}\n'

training_h = all_good_h[:1000]
testing_h = all_good_h[1001:]
training_h[:100]

prompt = 'Take a deep breath and work on this problem step-by-step. You are a mathematical tool to predict some model. Your job is to predict B for given A. The following is the dataset that you can use for the prediction.\n'
for i in range(len(training_h)):
  prompt += f'If A is {training_h[i][0], training_h[i][1], training_h[i][2], training_h[i][3]} then B is {training_h[i][4]}\n'

print(prompt)

class PredictdModel(BaseModel):
  predict: int = Field(..., description="The predicted B value")

llm = ChatOllama(model="bigllama/mistralv01-7b", temperature=0.2)
structured_demand = llm.with_structured_output(PredictdModel)

accuracy_count = 0

for item_h in testing_h:
  hpp_val = item_h[0]
  h_val_for_sinr_denominator = item_h[1]
  hps_val = item_h[2]
  hss_val = item_h[3]
  original_j = item_h[4]

  resp = structured_demand.invoke([
    SystemMessage(content=prompt),
    HumanMessage(
        content=f'If A is {hpp_val}, {h_val_for_sinr_denominator}, {hps_val}, {hss_val} then B is'
    )
  ])

  sinr = hpp_val * 100 / (1.0 + h_val_for_sinr_denominator * resp.predict)

  if sinr <= 70:
    print(f'If A is {(hpp_val, h_val_for_sinr_denominator, hps_val, hss_val)} then B is {resp.predict} GOOD, Original one is {original_j}\n')
    accuracy_count += 1
  else:
    print(f'If A is {(hpp_val, h_val_for_sinr_denominator, hps_val, hss_val)} then B is {resp.predict} BAD, Original one is {original_j}\n')

print(f'Accuracy: {accuracy_count/len(testing_h)}')