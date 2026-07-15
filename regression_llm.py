import random
from click import Tuple
import numpy as np
import math

repeated = set()
data = []

f = 2e9
c = 3e8
wave = c / f  # This equals 0.15 meters
P_max = 100.0

# Adjust this threshold to shift the average P2 up or down
I_max = 1500.0 

# We scale the tiny physics numbers (like 0.00000005) up by 10^8 
# so they look like normal numbers (e.g., 5.0 to 60.0) for the LLM.
scale_factor = 1e8

while len(data) < 1000:
    # 1. Generate Coordinates
    txp = [random.uniform(0, 10), random.uniform(26, 50)]
    txs = [random.uniform(0, 10), random.uniform(0, 25)]
    rxp = [random.uniform(15, 50), random.uniform(0, 25)]
    rxs = [random.uniform(15, 50), random.uniform(26, 50)]

    # 2. Calculate Distances
    dpp = np.sqrt((txp[0]-rxp[0])**2 + (txp[1]-rxp[1])**2)
    dsp = np.sqrt((txs[0]-rxp[0])**2 + (txs[1]-rxp[1])**2)
    dps = np.sqrt((txp[0]-rxs[0])**2 + (txp[1]-rxs[1])**2)
    dss = np.sqrt((txs[0]-rxs[0])**2 + (txs[1]-rxs[1])**2)

    # 3. Calculate True Physics Path Gain: (wavelength / (4 * pi * distance))^2
    hpp = (wave / (4 * np.pi * dpp))**2
    hsp = (wave / (4 * np.pi * dsp))**2
    hps = (wave / (4 * np.pi * dps))**2
    hss = (wave / (4 * np.pi * dss))**2

    # 4. Normalize to readable continuous floats (e.g., 42.15)
    hpp_normal = round(hpp * scale_factor, 2)
    hsp_normal = round(hsp * scale_factor, 2)
    hps_normal = round(hps * scale_factor, 2)
    hss_normal = round(hss * scale_factor, 2)

    # Check for duplicates using the full tuple of normalized values
    tuple_data = (hpp_normal, hsp_normal, hps_normal, hss_normal)
    if tuple_data in repeated:
        continue
    repeated.add(tuple_data)

    # 5. Calculate optimal P2 directly from the normalized inputs
    # If interference channel is 0 (impossible here, but good practice), max it out
    if hsp_normal <= 0:
        best_P2 = P_max
    else:
        allowed_p2 = I_max / hsp_normal
        best_P2 = round(min(allowed_p2, P_max), 2)

    # Save to dataset
    print(f"Generated Data Point: Hpp: {hpp_normal}, Hsp: {hsp_normal}, Hps: {hps_normal}, Hss: {hss_normal}, Best P2: {best_P2}")
    data.append([hpp_normal, hsp_normal, hps_normal, hss_normal, best_P2])
    # print(f"Best P1: {best_P1}, Best P2: {best_P2}, Best Fair Score: {best_fair_score}")

    
# train_data = data[:800]
# test_data = data[800:]

# prompt = 'You are a mathematical tool. Take a deep breath and predict B for given A.\n'
# for row in train_data:
#     prompt += f'Given A: {row[:-1]}, predict B: {row[-1]}\n'
# print(prompt)

# class PredictdModel(BaseModel):
#     predict: Tupleint = Field(..., description="The predicted B value")

# llm = ChatOllama(model="bigllama/mistralv01-7b", temperature=0.0)
# structured_demand = llm.with_structured_output(PredictdModel)
# MAE_llm = 0
# for row in test_data:
#     resp = structured_demand.invoke([
#     SystemMessage(content=prompt),
#     HumanMessage(content=f"Given A: {row[:-1]}, predict B: {row[-1]}\n")
#     ])

#     P2_llm = resp.predict
#     MAE_llm += abs(P2_llm - row[-1])

# MAE_llm /= len(test_data)
# print(f"MAE for LLM: {MAE_llm}")