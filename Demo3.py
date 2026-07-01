# This one we use the same paper code, let's see what going to happen

from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
import random
from langchain.chat_models import init_chat_model
import numpy as np


SIZE_AREA = 50.0
D2D_DIST = 10.0
PL_ALPHA = 38.0
PL_CONST = 34.5
TX_MAX = 100.0
BW = 1e7
NOISE = BW * 10**(-17.4)
I_THR = 10**(-55.0/10)
DUE_THR = 4.0
NUM_SAMPLES = 2000
GRID_RES = 100
TRAIN_SIZE = 50

np.random.seed(42)

def generate_one_channel():
    st_loc = SIZE_AREA * (np.random.rand(2) - 0.5)
    delta = 2 * D2D_DIST * (np.random.rand(2) - 0.5)
    sr_loc = st_loc + delta
    while (np.max(np.abs(sr_loc)) > SIZE_AREA / 2) or (np.linalg.norm(delta) > D2D_DIST):
        delta = 2 * D2D_DIST * (np.random.rand(2) - 0.5)
        sr_loc = st_loc + delta
    pt_loc = SIZE_AREA * (np.random.rand(2) - 0.5)
    pr_loc = np.array([0.0, 0.0])
    tx_locs = np.vstack([st_loc, pt_loc])
    rx_locs = np.vstack([sr_loc, pr_loc])
    dist_vec = rx_locs.reshape(2, 1, 2) - tx_locs
    dist_mat = np.linalg.norm(dist_vec, axis=2)
    dist_mat = np.maximum(dist_mat, 3.0)
    pl_db = -PL_CONST - PL_ALPHA * np.log10(dist_mat)
    pl_lin = 10 ** (pl_db / 10.0)
    fading = 0.5 * np.random.randn(2, 2)**2 + 0.5 * np.random.randn(2, 2)**2
    ch = np.maximum(pl_lin * fading, np.exp(-30))
    return ch[1, 1], ch[1, 0], ch[0, 1], ch[0, 0]

def generate_dataset_v2(n, res=GRID_RES, Pp=TX_MAX):
    Ps = np.linspace(0, TX_MAX, res)
    out = np.zeros((n, 5))
    for i in range(n):
        hpp, hsp, hps, hss = generate_one_channel()
        rate = np.log2(1.0 + Ps * hss / (Pp * hps + NOISE))
        ok = (Ps * hsp <= I_THR) & (rate >= DUE_THR)
        obj = rate.copy()
        obj[~ok] = -np.inf
        idx = np.argmax(obj)
        out[i] = [hpp, hsp, hps, hss, Ps[idx] if np.isfinite(obj[idx]) else 0.0]
    return out

data_v2 = generate_dataset_v2(NUM_SAMPLES)

h_log = np.log(data_v2[:, :4])
chan_avg = np.mean(h_log)
chan_std = np.std(h_log)
data_norm = np.column_stack([(h_log - chan_avg) / chan_std * 100.0, data_v2[:, 4]])

train = data_norm[:TRAIN_SIZE]
test = data_norm[TRAIN_SIZE:]
test_raw = data_v2[TRAIN_SIZE:]

prompt = 'Take a deep breath and work on this problem step-by-step. You are a mathematical tool to predict some model. Your job is to predict B for given A. The following is the dataset that you can use for the prediction.\n'
for i in range(len(train)):
    a0, a1, a2, a3, b = train[i]
    prompt += f'If A is {a0:.0f}, {a1:.0f}, {a2:.0f}, {a3:.0f}, then B is {b:.0f}.\n'

class PredictdModel(BaseModel):
  predict: int = Field(..., description="The predicted B value")

llm = ChatOllama(model="bigllama/mistralv01-7b", temperature=0.2)
structured_demand = llm.with_structured_output(PredictdModel)

accuracy_count = 0
for j in range(len(test)):
    a0, a1, a2, a3, _ = test[j]
    hpp_val, hsp_val, hps_val, hss_val, original_j = test_raw[j]
    current_channels = f'{a0:.0f}, {a1:.0f}, {a2:.0f}, {a3:.0f}'
    
    resp = structured_demand.invoke([
      SystemMessage(content=prompt),
      HumanMessage(
          content=f'If A is {hpp_val}, {h_val_for_sinr_denominator}, {hps_val}, {hss_val} then B is'
      )
    ])
    text = resp["choices"][0]["text"].strip()
    
    try:
        Ps_pred = float(resp.predict)
    except (ValueError, IndexError):
        Ps_pred = 0.0
    
    sinr = Ps_pred * hsp_val
    
    if sinr <= I_THR:
        print(f'If A is {(hpp_val, hsp_val, hps_val, hss_val)} then B is {Ps_pred} GOOD, Original one is {original_j}\n')
        accuracy_count += 1
    else:
        print(f'If A is {(hpp_val, hsp_val, hps_val, hss_val)} then B is {Ps_pred} BAD, Original one is {original_j}\n')

print(f'Accuracy: {accuracy_count/len(test)}')