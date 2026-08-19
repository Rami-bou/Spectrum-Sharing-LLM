import os
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
from graph import app
from environment import primary_I_max, calculate_secondary_discrete_rate, gen_channels, MCS, M, train, test, calculate_primary_discrete_rate, get_mcs_threshold

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULT_DIR = os.path.join("results", f"baseline_{timestamp}")
os.makedirs(RESULT_DIR, exist_ok=True)

def save_file(metrics_path):
    with open(metrics_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BASELINE PERFORMANCE\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Average Secondary Rate      : {np.mean(se_pred_list):.2f} Mbps\n")
        f.write(f"Average Optimal Rate        : {np.mean(se_true_list):.2f} Mbps\n")

        efficiency = 100 * np.mean(se_pred_list) / np.mean(se_true_list)

        f.write(f"Efficiency                 : {efficiency:.2f} %\n\n")

        f.write(f"Average Interference        : {np.mean(interf_pred_list):.2f}\n")
        f.write(f"Maximum Interference        : {np.max(interf_pred_list):.2f}\n")

        violation_rate = 100 * np.mean(violation_list)

        f.write(f"Violation Rate             : {violation_rate:.2f} %\n\n")

        #f.write(f"Average Negotiation Rounds : {np.mean(rounds_list):.2f}\n")

        success_rate = 100 * np.mean(success_list)

        f.write(f"Negotiation Success Rate   : {success_rate:.2f} %\n")

import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

# --- PRE-SETUP ---
POISON_FACTOR = 0.1

def compute_worst_margin(P1_vector, P2_vector, direct_h_primary, cross_h_primary):
    total_p2 = sum(P2_vector)
    margins = []
    for j in range(len(P1_vector)):
        signal = P1_vector[j] * direct_h_primary[j]
        if signal <= 0: continue
        baseline_sinr_db = 10 * math.log10(signal)
        target_th = get_mcs_threshold(baseline_sinr_db)
        if target_th < 0: continue
        interference = total_p2 * cross_h_primary[j]
        actual_sinr_linear = signal / (1.0 + interference)
        actual_sinr_db = 10 * math.log10(actual_sinr_linear) if actual_sinr_linear > 0 else -999
        margins.append(actual_sinr_db - target_th)
    return min(margins) if margins else -999.0

# Metric Lists
se_pred_list_sec = []
se_true_list_sec = []

se_pred_list_primary_ACTUAL = []  # The real damaged primary rate
se_true_list_primary = []

interf_pred_list_ACTUAL = []      # The real physical interference causing damage
interf_true_list = []

success_list = []
violation_list_ACTUAL = []        # Did the attack actually violate the physical constraint?

csv_path = os.path.join(RESULT_DIR, "benchmark_poisoned.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Sample", "TrueSecRate", "PredSecRate",
    "TruePrimaryRate", "ActualDamagedPrimaryRate",
    "TrueInterference", "ActualInjectedInterference", "ActualViolation", "Rounds", "Decision", "TrueP2", "PredP2"
])

print(f"\nStarting CSI Poisoning Benchmark over {len(test)} Test Samples...")

# --- MAIN LOOP ---
for i in range(len(test)):
    direct_h_prim = test[i][0]
    direct_h_sec = test[i][1]
    cross_h_prim = test[i][2]      # TRUE cross channel (Physical reality)
    cross_h_sec = test[i][3]
    true_p1 = test[i][4]
    true_p2 = test[i][5]

    # 1. Create the False Reality (The Attack)
    poisoned_cross_h_pri = [max(1, v * POISON_FACTOR) for v in cross_h_prim]

    initial_state = {
        "direct_primary_channels": test[i][0],
        "direct_secondary_channels": test[i][1],
        "cross_primary_channels": poisoned_cross_h_pri, # Agent is fed the LIES
        "cross_secondary_channels": test[i][3],
        "P1": test[i][4],
        "P2": [0] * M,
        "primary_critique": "",
        "secondary_critique": "",
        "primary_decision": "",
        "delta_hist": [],
        "iteration": 0
    }

    # Agent negotiates based on poisoned channels
    result = app.invoke(initial_state)
    pred_p2 = result['P2']

    # 2. Evaluate Secondary Rate (Secondary doesn't care about Primary's channels)
    rate_pred_sec = calculate_secondary_discrete_rate(true_p1, pred_p2, direct_h_sec, cross_h_sec)
    rate_true_sec = calculate_secondary_discrete_rate(true_p1, true_p2, direct_h_sec, cross_h_sec)
    se_pred_list_sec.append(rate_pred_sec)
    se_true_list_sec.append(rate_true_sec)

    # 3. Evaluate Primary Rate (THE IMPACT)
    # -> Use TRUE channels here to see how much the throughput crashed
    rate_actual_primary = calculate_primary_discrete_rate(true_p1, pred_p2, direct_h_prim, cross_h_prim)
    rate_true_primary = calculate_primary_discrete_rate(true_p1, true_p2, direct_h_prim, cross_h_prim)
    se_pred_list_primary_ACTUAL.append(rate_actual_primary)
    se_true_list_primary.append(rate_true_primary)

    # 4. Evaluate Interference (THE IMPACT)
    # What the primary BELIEVED the interference was (why it accepted):
    interf_believed = sum(pred_p2) * max(poisoned_cross_h_pri) 
    
    # What the interference ACTUALLY is in the physical world (The damage):
    interf_actual = sum(pred_p2) * max(cross_h_prim) 
    interf_true = sum(true_p2) * max(cross_h_prim)
    
    interf_pred_list_ACTUAL.append(interf_actual)
    interf_true_list.append(interf_true)

    # 5. Check Violations 
    actual_violation = 1 if interf_actual > primary_I_max else 0
    violation_list_ACTUAL.append(actual_violation)
    success_list.append(1 if result["primary_decision"] == "ACCEPT" else 0)

    csv_writer.writerow([
        i + 1, rate_true_sec, rate_pred_sec,
        rate_true_primary, rate_actual_primary,
        interf_true, interf_actual, actual_violation,
        result["iteration"], result["primary_decision"], sum(true_p2), sum(pred_p2)
    ])

    print(f"Sample {i+1} | Sec Rate: {rate_pred_sec} | Primary Believed Interf: {interf_believed:.1f} | ACTUAL Injected Interf: {interf_actual:.1f} | Violation: {'YES' if actual_violation else 'NO'}")

csv_file.close()

# --- CONSOLE METRICS ---
print(f"\n=== System Benchmark AFTER Attack ===")
print(f"Average Secondary Rate (Agent): {np.mean(se_pred_list_sec):.2f} Mbps")
print(f"Average Primary Rate (Actual Damaged): {np.mean(se_pred_list_primary_ACTUAL):.2f} Mbps")
print(f"Average Interference (Actual Injected): {np.mean(interf_pred_list_ACTUAL):.2f}")
print(f"Max Interference (Actual Injected): {np.max(interf_pred_list_ACTUAL):.2f}")
print(f"Agent Success Rate (Tricked Primary): {np.mean(success_list):.0%}")
print(f"ACTUAL Constraint Violations: {np.mean(violation_list_ACTUAL):.0%}")

# --- PLOTTING ---
if len(test) > 0:
    bin_size = 5
    num_bins = (len(test) + bin_size - 1) // bin_size
    bin_x = [i * bin_size for i in range(1, num_bins + 1)]

    # Binned Interference
    binned_interf_actual = [np.mean(interf_pred_list_ACTUAL[i : i + bin_size]) for i in range(0, len(interf_pred_list_ACTUAL), bin_size)]
    binned_interf_true = [np.mean(interf_true_list[i : i + bin_size]) for i in range(0, len(interf_true_list), bin_size)]

    # Plot 1: The Interference Damage
    plt.figure(figsize=(10, 5))
    plt.axhline(y=primary_I_max, color='black', linestyle='-', linewidth=2, label=f'Primary Limit (${{I_{{max}}}}={primary_I_max}$)')
    plt.plot(bin_x, binned_interf_true, label='Safe Optimal Interference', color='blue', linestyle='--', marker='o', linewidth=2)
    
    # Plotting the ACTUAL interference in RED to show the massive violation
    plt.plot(bin_x, binned_interf_actual, label='ACTUAL Attack Interference', color='red', linestyle='-', marker='x', linewidth=2, markersize=8)

    plt.title('Impact of CSI Poisoning: Real Physical Interference vs Limit', fontsize=13, color='darkred')
    plt.xlabel('Test Sample Index', fontsize=11)
    plt.ylabel('Average Max Interference Injected', fontsize=11)
    plt.xticks(bin_x)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "attack_interference_impact.png"), dpi=300)

    # Plot 2: Primary Throughput Damage
    binned_se_pred_primary = [np.mean(se_pred_list_primary_ACTUAL[i : i + bin_size]) for i in range(0, len(se_pred_list_primary_ACTUAL), bin_size)]
    binned_se_true_primary = [np.mean(se_true_list_primary[i : i + bin_size]) for i in range(0, len(se_true_list_primary), bin_size)]

    plt.figure(figsize=(10, 5))
    plt.plot(bin_x, binned_se_true_primary, label='True Optimal Primary Rate', color='blue', linestyle='--', marker='o')
    plt.plot(bin_x, binned_se_pred_primary, label='Damaged Primary Rate (Under Attack)', color='red', linestyle='-', marker='s')

    plt.title('Impact of CSI Poisoning: Primary Network Throughput Crash', fontsize=13, color='darkred')
    plt.xlabel('Test Sample Index', fontsize=11)
    plt.ylabel('Actual Primary Rate (Mbps)', fontsize=11)
    plt.xticks(bin_x)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "attack_primary_rate_crash.png"), dpi=300)

    plt.show()

# data = gen_channels(100)
# test_data = data[90:100]

# # 2. Define the prompt sizes to sweep
# train_sizes = [10, 30, 50, 70, 90]

# # Dictionaries/Lists to store results for our plots
# avg_rate_pred_list = []
# avg_rate_true_list = []
# interference_pred_dict = {size: [] for size in train_sizes}
# interference_true_dict = {size: [] for size in train_sizes}

# print("\nStarting Few-Shot Training Size Sweep...")

# for size in train_sizes:
#     print(f"\n{'='*40}")
#     print(f" EVALUATING WITH {size} TRAINING SAMPLES ")
#     print(f"{'='*40}")
    
#     train_data = data[:size]
    
#     # Update the global prompt variable used inside the `secondary` node
#     global prompt_secondary_allocation
#     prompt_secondary_allocation = build_prompt(train_data)
    
#     current_size_se_pred = []
#     current_size_se_true = []
    
#     for i in range(len(test_data)):
#         direct_h_sec = test_data[i][1] 
#         cross_h_sec = test_data[i][3]  
#         true_p1 = test_data[i][4]      
#         true_p2 = test_data[i][5]
#         cross_h_prim = test_data[i][2]
        
#         initial_state = {
#             "direct_primary_channels": test_data[i][0],
#             "direct_secondary_channels": test_data[i][1],
#             "cross_primary_channels": test_data[i][2],
#             "cross_secondary_channels": test_data[i][3],
#             "P1": test_data[i][4],                      
#             "P2": [0] * M,
#             "primary_critique": "",
#             "primary_decision": "",
#             "delta_hist": [],
#             "iteration": 0
#         }

#         # Run the LangGraph workflow
#         result = app.invoke(initial_state)
#         pred_p2 = result['P2']
        
#         # --- METRIC 1: Calculate Secondary Rates ---
#         se_pred = calculate_secondary_discrete_rate(true_p1, pred_p2, direct_h_sec, cross_h_sec)
#         se_true = calculate_secondary_discrete_rate(true_p1, true_p2, direct_h_sec, cross_h_sec)
#         current_size_se_pred.append(se_pred)
#         current_size_se_true.append(se_true)
        
#         # --- METRIC 2: Calculate Max Interference on Primary ---
#         # Interference caused to a primary receiver = total_P2 * cross_h_primary
#         max_interf_pred = sum(pred_p2) * max(cross_h_prim)
#         max_interf_true = sum(true_p2) * max(cross_h_prim)
        
#         interference_pred_dict[size].append(max_interf_pred)
#         interference_true_dict[size].append(max_interf_true)
        
#         print(f"Sample {i+1}: True P2 Sum = {sum(true_p2)}, Pred P2 Sum = {sum(pred_p2)}")

#     # Store the average rates for this training size
#     avg_rate_pred_list.append(np.mean(current_size_se_pred))
#     avg_rate_true_list.append(np.mean(current_size_se_true))

# # PLOT 1: Secondary Rate vs Training Size (Line Chart)
# plt.figure(figsize=(8, 6))
# plt.plot(train_sizes, avg_rate_true_list, label='True Optimal (Baseline)', color='blue', linestyle='--', marker='o')
# plt.plot(train_sizes, avg_rate_pred_list, label='LLM Agent Prediction', color='red', linestyle='-', marker='s')

# plt.title('Impact of Few-Shot Examples on Secondary Sum Rate', fontsize=14)
# plt.xlabel('Number of Training Samples in Prompt', fontsize=12)
# plt.ylabel('Average Secondary Sum Rate (Mbps)', fontsize=12)
# plt.xticks(train_sizes)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle=':', alpha=0.7)
# plt.tight_layout()
# plt.savefig("Result_Rate_vs_TrainSize.png")


# # PLOT 2: Primary Interference vs Training Size (Grouped Scatter)
# plt.figure(figsize=(10, 6))

# # Plot the hard threshold
# plt.axhline(y=1000, color='black', linestyle='-', linewidth=2, label='Primary Interference Limit ($I_{max}=1000$)')

# # We use a slight offset on the x-axis so True and Pred dots don't overlap completely
# offset = 1.5 
# for size in train_sizes:
#     x_true = [size - offset] * len(test_data)
#     x_pred = [size + offset] * len(test_data)
    
#     # Plot individual scatters for each test sample
#     plt.scatter(x_true, interference_true_dict[size], color='blue', alpha=0.5, marker='o', s=40)
#     plt.scatter(x_pred, interference_pred_dict[size], color='red', alpha=0.6, marker='x', s=40)

# # Add dummy scatter points just to populate the legend cleanly
# plt.scatter([], [], color='blue', alpha=0.5, marker='o', label='True Optimal Interference')
# plt.scatter([], [], color='red', alpha=0.6, marker='x', label='LLM Agent Interference')

# plt.title('Constraint Satisfaction: Primary Interference vs. Prompt Size', fontsize=14)
# plt.xlabel('Number of Training Samples in Prompt', fontsize=12)
# plt.ylabel('Max Caused Interference to Primary', fontsize=12)
# plt.xticks(train_sizes)
# plt.legend(fontsize=11, loc='upper right')
# plt.grid(True, linestyle=':', alpha=0.7)
# plt.tight_layout()
# plt.savefig("Result_Interference_vs_TrainSize.png")

# plt.show()
