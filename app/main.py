from dataset import gen_channels, MCS, M
import math
import numpy as np
from graph import app
import matplotlib.pyplot as plt

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

data = gen_channels(120)
train = data[:90]
test = data[90:100]
prompt_secondary_allocation = build_prompt(train)
all_pred_P2 = []
all_true_P2 = []
se_pred_list = []
se_true_list = []

def get_discrete_rate(sinr_linear):
    """Converts linear SINR to dB and maps it to a discrete data rate."""
    if sinr_linear <= 0:
        return 0
    
    sinr_db = 10 * math.log10(sinr_linear)
    
    achieved_rate = 0
    for threshold, rate in MCS:
        if sinr_db >= threshold:
            achieved_rate = rate
        else:
            break
            
    return achieved_rate

def calculate_primary_discrete_rate(P1_vector, P2_vector, direct_h_primary, cross_h_primary):
    """Calculates total Primary Throughput based on discrete MCS levels."""
    total_throughput_mbps = 0
    total_P2 = sum(P2_vector)
    
    for j in range(len(P1_vector)):
        signal = P1_vector[j] * direct_h_primary[j]
        interference_from_secondary = total_P2 * cross_h_primary[j]
        
        # Calculate physical linear SINR
        sinr_linear = signal / (1.0 + interference_from_secondary)
        
        # Map to discrete hardware throughput
        total_throughput_mbps += get_discrete_rate(sinr_linear)
        
    return total_throughput_mbps

print("\nStarting Benchmark over Test Dataset...")
for i in range(len(test)):
    direct_h_pri = test[i][0] 
    cross_h_pri = test[i][2]  
    true_p1 = test[i][4]      
    true_p2 = test[i][5] 
    
    initial_state = {
        "direct_primary_channels": test[i][0],
        "direct_secondary_channels": test[i][1],
        "cross_primary_channels": test[i][2],
        "cross_secondary_channels": test[i][3],
        "P1": test[i][4],                      
        "P2": [0] * M,
        "primary_critique": "",
        "primary_decision": "",
        "delta_hist": [],
        "iteration": 0
    }

    result = app.invoke(initial_state)
    
    pred_p2 = result['P2']
    
    all_pred_P2.append(pred_p2)
    all_true_P2.append(true_p2)
    
    se_pred_list.append(calculate_primary_discrete_rate(true_p1, pred_p2, direct_h_pri, cross_h_pri))
    se_true_list.append(calculate_primary_discrete_rate(true_p1, true_p2, direct_h_pri, cross_h_pri))

    print(f"Allocation P2 pred: {result['P2']}")
    print(f"Allocation P2 true: {test[i][5]}")

all_pred_P2 = np.array(all_pred_P2) 
all_true_P2 = np.array(all_true_P2) 

mae_per_receiver = np.mean(np.abs(all_pred_P2 - all_true_P2), axis=0)

print("\n" + "="*40)
print(" BENCHMARK RESULTS: MEAN ABSOLUTE ERROR ")
print("="*40)
for j in range(len(mae_per_receiver)):
    print(f"Secondary Receiver {j+1} MAE: {mae_per_receiver[j]:.2f} Watts")
print("="*40 + "\n")

window_size = 5 if len(test) < 50 else 10

def moving_average(data, w):
    """Calculates the moving average shifting by 1 step at a time."""
    return np.convolve(data, np.ones(w), 'valid') / w

smoothed_se_pred = moving_average(se_pred_list, window_size)
smoothed_se_true = moving_average(se_true_list, window_size)

plt.figure(figsize=(12, 6))
plt.plot(smoothed_se_true, label=f'True Optimal Primary SE', color='blue', linestyle='--', marker='o', markersize=4)
plt.plot(smoothed_se_pred, label=f'Agent-Protected Primary SE', color='red', linestyle='-', marker='s', markersize=4)

plt.title(f'Primary Network Spectral Efficiency Comparison\n(Moving Average, Window={window_size})', fontsize=14)
plt.xlabel('Test Sample Index (Rolling Window)', fontsize=12)
plt.ylabel('Sum Spectral Efficiency (bps/Hz)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("Result_MCS.png")
plt.show()