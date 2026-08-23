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


# POISON_FACTOR = 0.1


# def compute_worst_margin(P1_vector, P2_vector, direct_h_primary, cross_h_primary):
#     total_p2 = sum(P2_vector)
#     margins = []
#     for j in range(len(P1_vector)):
#         signal = P1_vector[j] * direct_h_primary[j]
#         if signal <= 0: continue
#         baseline_sinr_db = 10 * math.log10(signal)
#         target_th = get_mcs_threshold(baseline_sinr_db)
#         if target_th < 0: continue
#         interference = total_p2 * cross_h_primary[j]
#         actual_sinr_linear = signal / (1.0 + interference)
#         actual_sinr_db = 10 * math.log10(actual_sinr_linear) if actual_sinr_linear > 0 else -999
#         margins.append(actual_sinr_db - target_th)
#     return min(margins) if margins else -999.0

se_pred_list = []
se_true_list = []

interf_pred_list = []
interf_true_list = []

rounds_list = []
success_list = []
violation_list = []
se_pred_list_primary = []
se_true_list_primary = []

csv_path = os.path.join(RESULT_DIR, "benchmark.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
"Sample", "TrueRate", "PredRate",
"TruePrimaryRate", "PredPrimaryRate",
"TrueInterference", "PredInterference", "Violation", "Rounds", "Decision", "TrueP2", "PredP2"
])

# to see the clifffffffffffff
sinr_csv_path = os.path.join(RESULT_DIR, "sinr_analysis.csv")
sinr_csv_file = open(sinr_csv_path, "w", newline="")
sinr_writer = csv.writer(sinr_csv_file)
sinr_writer.writerow([
    "Sample", "Receiver", "Target_Cliff_dB", "Baseline_SINR_dB",
    "True_SINR_dB", "True_Margin_dB", "Pred_SINR_dB", "Pred_Margin_dB",
    "True_P2_Sum", "Pred_P2_Sum", "Crash_Flag"
])

print(f"\nStarting Benchmark over {len(test)} Test Samples...")

for i in range(len(test)):
    direct_h_sec = test[i][1]
    cross_h_sec = test[i][3]
    direct_h_prim = test[i][0]
    cross_h_prim = test[i][2]
    true_p1 = test[i][4]
    true_p2 = test[i][5]

    # poisoned_cross_h_pri = [max(1, v * POISON_FACTOR) for v in cross_h_prim]

    initial_state = {
    "direct_primary_channels": test[i][0],
    "direct_secondary_channels": test[i][1],
    "cross_primary_channels": test[i][2],
    "cross_secondary_channels": test[i][3],
    "P1": test[i][4],
    "P2": [0] * M,
    "primary_critique": "",
    "secondary_critique": "",
    "primary_decision": "",
    "delta_hist": [],
    "iteration": 0
    }

    result = app.invoke(initial_state)

    pred_p2 = result['P2']

    for j in range(len(direct_h_prim)):
        p1_val = true_p1[j] if isinstance(true_p1, (list, np.ndarray)) else true_p1
        signal = p1_val * direct_h_prim[j]
        
        if signal <= 0:
            continue
            
        baseline_sinr_db = 10 * math.log10(signal)
        target_th = get_mcs_threshold(baseline_sinr_db)
        
        # True SINR & Margin
        true_interf = sum(true_p2) * cross_h_prim[j]
        true_sinr_lin = signal / (1.0 + true_interf)
        true_sinr_db = 10 * math.log10(true_sinr_lin) if true_sinr_lin > 0 else -999.0
        true_margin = true_sinr_db - target_th
        
        # Predicted SINR & Margin
        pred_interf = sum(pred_p2) * cross_h_prim[j]
        pred_sinr_lin = signal / (1.0 + pred_interf)
        pred_sinr_db = 10 * math.log10(pred_sinr_lin) if pred_sinr_lin > 0 else -999.0
        pred_margin = pred_sinr_db - target_th
        
        crash_flag = 1 if pred_margin < 0 else 0
        
        sinr_writer.writerow([
            i + 1,
            j + 1,
            round(target_th, 2),
            round(baseline_sinr_db, 2),
            round(true_sinr_db, 2),
            round(true_margin, 2),
            round(pred_sinr_db, 2),
            round(pred_margin, 2),
            sum(true_p2),
            sum(pred_p2),
            crash_flag
        ])

    print(f"If {test[i][0]} {test[i][1]} {test[i][2]} {test[i][3]} then pred {pred_p2}, true {test[i][5]}")

    for j in range(len(direct_h_prim)):
        # Handle whether true_p1 is a list/vector or a single scalar
        p1_val = true_p1[j] if isinstance(true_p1, (list, np.ndarray)) else true_p1
        signal = p1_val * direct_h_prim[j]
        
        if signal <= 0:
            continue
            
        # Baseline SINR in dB (when P2 = 0)
        baseline_sinr_db = 10 * math.log10(signal)
        
        # Target MCS cliff threshold for this receiver
        target_th = get_mcs_threshold(baseline_sinr_db)
        if target_th < 0:
            continue
            
        # 1. Actual SINR with True Optimal P2
        true_interference = sum(true_p2) * cross_h_prim[j]
        true_sinr_linear = signal / (1.0 + true_interference)
        true_sinr_db = 10 * math.log10(true_sinr_linear) if true_sinr_linear > 0 else -999
        
        # 2. Actual SINR with LLM Predicted P2
        pred_interference = sum(pred_p2) * cross_h_prim[j]
        pred_sinr_linear = signal / (1.0 + pred_interference)
        pred_sinr_db = 10 * math.log10(pred_sinr_linear) if pred_sinr_linear > 0 else -999
        
        # 3. Calculate margin (how close the LLM is to breaking the primary receiver)
        margin_to_cliff = pred_sinr_db - true_sinr_db
        
        print(f"  RX {j+1} | Target Cliff: {target_th:>5.2f} dB | True SINR: {true_sinr_db:>5.2f} dB | Pred SINR: {pred_sinr_db:>5.2f} dB | Margin: {margin_to_cliff:>6.2f} dB")
        
        if margin_to_cliff < 0:
            print(f"    >>> CRASH DETECTED: LLM interference dropped RX {j+1} below the MCS threshold!")
            
    print("-" * 40)

    # margin_believed = compute_worst_margin(true_p1, pred_p2, direct_h_prim, poisoned_cross_h_pri)
    # margin_actual = compute_worst_margin(true_p1, pred_p2, direct_h_prim, cross_h_prim)

    # 1. Calculate Discrete Secondary Rates
    rate_pred = calculate_secondary_discrete_rate(true_p1, pred_p2, direct_h_sec, cross_h_sec)
    rate_true = calculate_secondary_discrete_rate(true_p1, true_p2, direct_h_sec, cross_h_sec)
    se_pred_list.append(rate_pred)
    se_true_list.append(rate_true)

    rate_pred_primary = calculate_primary_discrete_rate(true_p1, pred_p2, direct_h_prim, cross_h_prim)
    rate_true_primary = calculate_primary_discrete_rate(true_p1, true_p2, direct_h_prim, cross_h_prim)
    se_pred_list_primary.append(rate_pred_primary)
    se_true_list_primary.append(rate_true_primary)


    # 2. Calculate Worst-Case Caused Interference on Primary Receivers

    max_interf_pred = sum(pred_p2) * max(cross_h_prim)
    max_interf_true = sum(true_p2) * max(cross_h_prim)
    interf_pred_list.append(max_interf_pred)
    interf_true_list.append(max_interf_true)
    # 3. Calculate the sucess rate
    success_list.append(1 if result["primary_decision"] == "ACCEPT" else 0)
    # 4. Constraint violation
    violation_list.append(1 if max_interf_pred > primary_I_max else 0)

    csv_writer.writerow([
    i + 1, rate_true, rate_pred,
    rate_true_primary, rate_pred_primary,
    max_interf_true, max_interf_pred, violation_list[i],
    result["iteration"], result["primary_decision"], sum(true_p2), sum(pred_p2)
    ])

    print(f"Sample {i+1}/100 | True Rate: {rate_true} | Pred Rate: {rate_pred} | Pred Interf: {max_interf_pred:.1f}")
    print(f"True P2: {true_p2}")
    print(f"pred P2: {result['P2']}")


csv_file.close()

metrics_path = os.path.join(RESULT_DIR, "metrics.txt")

save_file(metrics_path)

print(f"System Benchmark Before attack:\n")
print(f"Average Secondary Rate (True): {np.mean(se_true_list):.2f}")
print(f"Average Secondary Rate (Predicted): {np.mean(se_pred_list):.2f}")
print(f"Average Interference (Predicted): {np.mean(interf_pred_list):.2f}")
print(f"Max Interference (Predicted): {np.max(interf_pred_list):.2f}")
print(f"Efficiency: {np.mean(success_list):.0%}")
print(f"Constraint Violations: {np.sum(violation_list):.0%}")

# Only attempt to plot if there are enough test samples for binning
if len(test) > 0:
    # Create an index for every individual test sample (1-based index)
    x = list(range(1, len(test) + 1))

    # --- 1. Secondary Rate Plot (Scatter) ---
    plt.figure(figsize=(10, 5))
    plt.scatter(x, se_true_list, label='True Optimal Secondary Rate', color='blue', alpha=0.7, marker='o', s=30)
    plt.scatter(x, se_pred_list, label='LLM Agent Secondary Rate', color='red', alpha=0.7, marker='s', s=30)

    plt.xlabel('Test Sample Index', fontsize=11)
    plt.ylabel('Secondary Rate (Mbps)', fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULT_DIR, "secondary_rate_csi_pois.png"),
        dpi=300,
        bbox_inches="tight"
    )

    # --- 2. Primary Rate Plot (Scatter) ---
    plt.figure(figsize=(10, 5))
    plt.scatter(x, se_true_list_primary, label='True Optimal Primary Rate', color='blue', alpha=0.7, marker='o', s=30)
    plt.scatter(x, se_pred_list_primary, label='LLM Agent Primary Rate', color='red', alpha=0.7, marker='s', s=30)

    plt.xlabel('Test Sample Index', fontsize=11)
    plt.ylabel('Primary Rate (Mbps)', fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULT_DIR, "attack_primary_rate_crash.png"),
        dpi=300,
        bbox_inches="tight"
    )

    # --- 3. Interference Plot (Scatter) ---
    plt.figure(figsize=(10, 5))
    plt.axhline(y=primary_I_max, color='black', linestyle='-', linewidth=2, label=f'Primary Interference Limit (I_max={primary_I_max})')
    plt.scatter(x, interf_true_list, label='True Optimal Interference', color='blue', alpha=0.7, marker='o', s=30)
    plt.scatter(x, interf_pred_list, label='LLM Agent Interference', color='red', alpha=0.7, marker='x', s=40)

    plt.xlabel('Test Sample Index', fontsize=11)
    plt.ylabel('Max Interference Injected', fontsize=11)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULT_DIR, "attack_interference_impact.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
else:
    print("No test samples to plot.")
# if len(test) > 0:
#     bin_size = 5
#     # Recalculate num_bins to ensure it's at least 1 if there's data, or correctly reflects the number of bins
#     num_bins = (len(test) + bin_size - 1) // bin_size
#     # Recalculate bin_x based on the corrected num_bins
#     bin_x = [i * bin_size for i in range(1, num_bins + 1)]
#     # The binned lists are already calculated correctly based on len(se_pred_list) and bin_size
#     binned_se_pred = [np.mean(se_pred_list[i : i + bin_size]) for i in range(0, len(se_pred_list), bin_size)]
#     binned_se_true = [np.mean(se_true_list[i : i + bin_size]) for i in range(0, len(se_true_list), bin_size)]

#     binned_se_pred_primary = [np.mean(se_pred_list_primary[i : i + bin_size]) for i in range(0, len(se_pred_list_primary), bin_size)]
#     binned_se_true_primary = [np.mean(se_true_list_primary[i : i + bin_size]) for i in range(0, len(se_true_list_primary), bin_size)]

#     binned_interf_pred = [np.mean(interf_pred_list[i : i + bin_size]) for i in range(0, len(interf_pred_list), bin_size)]
#     binned_interf_true = [np.mean(interf_true_list[i : i + bin_size]) for i in range(0, len(interf_true_list), bin_size)]

#     plt.figure(figsize=(10, 5))
#     plt.plot(bin_x, binned_se_true, label='True Optimal Secondary Rate', color='blue', linestyle='--', marker='o', linewidth=2)
#     plt.plot(bin_x, binned_se_pred, label='LLM Agent Secondary Rate', color='red', linestyle='-', marker='s', linewidth=2)

#     # plt.title('Secondary Network Sum Rate (Averaged Every 5 Test Samples)', fontsize=13)
#     plt.xlabel('Test Sample Index', fontsize=11)
#     plt.ylabel('Average Secondary Rate (Mbps)', fontsize=11)
#     plt.xticks(bin_x)
#     plt.legend(fontsize=11)
#     plt.grid(True, linestyle=':', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig(
#     os.path.join(
#     RESULT_DIR,
#     "secondary_rate_csi_pois.png"
#     ),
#     dpi=300,
#     bbox_inches="tight"
#     )

#     plt.figure(figsize=(10, 5))
#     plt.plot(bin_x, binned_se_true_primary, label='True Optimal Primary Rate', color='blue', linestyle='--', marker='o', linewidth=2)
#     plt.plot(bin_x, binned_se_pred_primary, label='LLM Agent Primary Rate', color='red', linestyle='-', marker='s', linewidth=2)
#     # plt.title('Primary Network Sum Rate (Averaged Every 5 Test Samples)', fontsize=13)
#     plt.xlabel('Test Sample Index', fontsize=11)
#     plt.ylabel('Average Primary Rate (Mbps)', fontsize=11)
#     plt.xticks(bin_x)
#     plt.legend(fontsize=11)
#     plt.grid(True, linestyle=':', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig(
#     os.path.join(
#     RESULT_DIR,
#     "attack_primary_rate_crash.png"
#     ),
#     dpi=300,
#     bbox_inches="tight"
#     )

#     plt.figure(figsize=(10, 5))
#     plt.axhline(y=primary_I_max, color='black', linestyle='-', linewidth=2, label=f'Primary Interference Limit (${{I_{{max}}}}={primary_I_max}$)')
#     plt.plot(bin_x, binned_interf_true, label='True Optimal Interference', color='blue', linestyle='--', marker='o', linewidth=2)
#     plt.plot(bin_x, binned_interf_pred, label='LLM Agent Interference', color='red', linestyle='-', marker='x', linewidth=2, markersize=8)
#     # plt.title('Primary Network Protection: Caused Interference (Averaged Every 5 Test Samples)', fontsize=13)
#     plt.xlabel('Test Sample Index', fontsize=11)
#     plt.ylabel('Average Max Interference Injected', fontsize=11)
#     plt.xticks(bin_x)
#     plt.legend(fontsize=11, loc='upper right')
#     plt.grid(True, linestyle=':', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig(
#     os.path.join(
#     RESULT_DIR,
#     "attack_interference_impact.png"
#     ),
#     dpi=300,
#     bbox_inches="tight"
#     )
#     plt.show()
# else:
#     print("No test samples to plot.") 

# import os
# import csv
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime

# import environment
# from environment import (
#     gen_channels,
#     primary_I_max,
#     calculate_secondary_discrete_rate,
#     calculate_primary_discrete_rate,
#     M,
#     get_mcs_threshold
# )
# from graph import app

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# RESULT_DIR = os.path.join("results", f"spatial_benchmark_{timestamp}")
# os.makedirs(RESULT_DIR, exist_ok=True)

# positions = [[20, 20], [30, 30], [40, 40], [50, 50], [60, 60], [70, 70]]
# position_labels = [f"[{x},{y}]" for x, y in positions]

# pos_avg_se_pred = []
# pos_avg_se_true = []
# pos_avg_prim_pred = []
# pos_avg_prim_true = []
# pos_avg_interf_pred = []
# pos_avg_interf_true = []
# pos_violation_rate = []
# pos_success_rate = []

# csv_path = os.path.join(RESULT_DIR, "spatial_benchmark.csv")
# csv_file = open(csv_path, "w", newline="")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow([
#     "Position", "Sample", "TrueRate", "PredRate",
#     "TruePrimaryRate", "PredPrimaryRate",
#     "TrueInterference", "PredInterference", "Violation", "Rounds", "Decision"
# ])

# print(f"Starting Spatial Benchmark across {len(positions)} positions...\n")

# for pos in positions:
#     print(f"==================================================")
#     print(f"Evaluating Secondary Position: {pos}")
#     print(f"==================================================")

#     # Clear global environment data list before generating fresh channels for the new location
#     environment.data.clear()
#     dataset = gen_channels(100, pos)
#     test_samples = dataset[70:]  # Evaluation subset (30 test samples per position)

#     se_pred_list = []
#     se_true_list = []
#     se_pred_list_primary = []
#     se_true_list_primary = []
#     interf_pred_list = []
#     interf_true_list = []
#     violation_list = []
#     success_list = []

#     for i, test in enumerate(test_samples):
#         direct_h_prim = test[0]
#         direct_h_sec = test[1]
#         cross_h_prim = test[2]
#         cross_h_sec = test[3]
#         true_p1 = test[4]
#         true_p2 = test[5]

#         initial_state = {
#             "direct_primary_channels": direct_h_prim,
#             "direct_secondary_channels": direct_h_sec,
#             "cross_primary_channels": cross_h_prim,
#             "cross_secondary_channels": cross_h_sec,
#             "P1": true_p1,
#             "P2": [0] * M,
#             "primary_critique": "",
#             "secondary_critique": "",
#             "primary_decision": "",
#             "delta_hist": [],
#             "iteration": 0
#         }

#         result = app.invoke(initial_state)
#         pred_p2 = result["P2"]

#         print(f"If {test[i][0]} {test[i][1]} {test[i][2]} {test[i][3]} then pred {pred_p2}, true {test[i][5]}")

#         for j in range(3):
#             signal = test[i][4] * test[i][0]
                
#             # Baseline SINR in dB (when P2 = 0)
#             baseline_sinr_db = 10 * math.log10(signal)
            
#             # Target MCS cliff threshold for this receiver
#             target_th = get_mcs_threshold(baseline_sinr_db)
#             if target_th < 0:
#                 continue
                
#             # Actual SINR in dB with current P2 proposal
#             interference = sum(test[i][5]) * test[i][2]
#             actual_sinr_linear = signal / (1.0 + interference)
#             actual_sinr_db = 10 * math.log10(actual_sinr_linear) if actual_sinr_linear > 0 else -999
            
#             print(f"SINR by true: {actual_sinr_db}")

#         for j in range(3):
#             signal = test[i][4] * test[i][0]
                
#             # Baseline SINR in dB (when P2 = 0)
#             baseline_sinr_db = 10 * math.log10(signal)
            
#             # Target MCS cliff threshold for this receiver
#             target_th = get_mcs_threshold(baseline_sinr_db)
#             if target_th < 0:
#                 continue
                
#             # Actual SINR in dB with current P2 proposal
#             interference = sum(pred_p2) * test[i][2]
#             actual_sinr_linear = signal / (1.0 + interference)
#             actual_sinr_db = 10 * math.log10(actual_sinr_linear) if actual_sinr_linear > 0 else -999
            
#             print(f"SINR by pred: {actual_sinr_db}")

#         rate_pred = calculate_secondary_discrete_rate(true_p1, pred_p2, direct_h_sec, cross_h_sec)
#         rate_true = calculate_secondary_discrete_rate(true_p1, true_p2, direct_h_sec, cross_h_sec)
#         se_pred_list.append(rate_pred)
#         se_true_list.append(rate_true)

#         rate_pred_primary = calculate_primary_discrete_rate(true_p1, pred_p2, direct_h_prim, cross_h_prim)
#         rate_true_primary = calculate_primary_discrete_rate(true_p1, true_p2, direct_h_prim, cross_h_prim)
#         se_pred_list_primary.append(rate_pred_primary)
#         se_true_list_primary.append(rate_true_primary)

#         max_interf_pred = sum(pred_p2) * max(cross_h_prim)
#         max_interf_true = sum(true_p2) * max(cross_h_prim)
#         interf_pred_list.append(max_interf_pred)
#         interf_true_list.append(max_interf_true)

#         is_violation = 1 if max_interf_pred > primary_I_max else 0
#         is_success = 1 if result["primary_decision"] == "ACCEPT" else 0
#         violation_list.append(is_violation)
#         success_list.append(is_success)

#         csv_writer.writerow([
#             str(pos), i + 1, rate_true, rate_pred,
#             rate_true_primary, rate_pred_primary,
#             max_interf_true, max_interf_pred, is_violation,
#             result["iteration"], result["primary_decision"]
#         ])

#         print(f"Pos {pos} | Sample {i+1}/{len(test_samples)} | Sec Rate: {rate_pred:.1f} Mbps | Interf: {max_interf_pred:.1f}")

#     pos_avg_se_pred.append(np.mean(se_pred_list))
#     pos_avg_se_true.append(np.mean(se_true_list))
#     pos_avg_prim_pred.append(np.mean(se_pred_list_primary))
#     pos_avg_prim_true.append(np.mean(se_true_list_primary))
#     pos_avg_interf_pred.append(np.mean(interf_pred_list))
#     pos_avg_interf_true.append(np.mean(interf_true_list))
#     pos_violation_rate.append(100 * np.mean(violation_list))
#     pos_success_rate.append(100 * np.mean(success_list))

# csv_file.close()

# # Save Aggregate Metrics Summary
# metrics_path = os.path.join(RESULT_DIR, "metrics.txt")
# with open(metrics_path, "w") as f:
#     f.write("=" * 60 + "\n")
#     f.write("SPATIAL BENCHMARK SUMMARY PER POSITION\n")
#     f.write("=" * 60 + "\n\n")
#     for idx, pos in enumerate(positions):
#         f.write(f"Position {pos}:\n")
#         f.write(f"  Secondary Rate (True / Pred) : {pos_avg_se_true[idx]:.2f} / {pos_avg_se_pred[idx]:.2f} Mbps\n")
#         f.write(f"  Primary Rate (True / Pred)   : {pos_avg_prim_true[idx]:.2f} / {pos_avg_prim_pred[idx]:.2f} Mbps\n")
#         f.write(f"  Interference (True / Pred)   : {pos_avg_interf_true[idx]:.2f} / {pos_avg_interf_pred[idx]:.2f}\n")
#         f.write(f"  Violation Rate               : {pos_violation_rate[idx]:.2f} %\n")
#         f.write(f"  Negotiation Success Rate     : {pos_success_rate[idx]:.2f} %\n\n")

# # Visualizations
# x_indices = np.arange(len(positions))

# # 1. Secondary Rate Plot
# plt.figure(figsize=(10, 5))
# plt.plot(x_indices, pos_avg_se_true, label='True Optimal Secondary Rate', color='blue', linestyle='--', marker='o', linewidth=2)
# plt.plot(x_indices, pos_avg_se_pred, label='LLM Agent Secondary Rate', color='red', linestyle='-', marker='s', linewidth=2)
# plt.xlabel('Secondary Network Position [x, y]', fontsize=11)
# plt.ylabel('Average Secondary Rate (Mbps)', fontsize=11)
# plt.xticks(x_indices, position_labels)
# plt.legend(fontsize=11)
# plt.grid(True, linestyle=':', alpha=0.7)
# plt.tight_layout()
# plt.savefig(os.path.join(RESULT_DIR, "secondary_rate_spatial.png"), dpi=300)

# # 2. Primary Rate Plot
# plt.figure(figsize=(10, 5))
# plt.plot(x_indices, pos_avg_prim_true, label='True Optimal Primary Rate', color='blue', linestyle='--', marker='o', linewidth=2)
# plt.plot(x_indices, pos_avg_prim_pred, label='LLM Agent Primary Rate', color='red', linestyle='-', marker='s', linewidth=2)
# plt.xlabel('Secondary Network Position [x, y]', fontsize=11)
# plt.ylabel('Average Primary Rate (Mbps)', fontsize=11)
# plt.xticks(x_indices, position_labels)
# plt.legend(fontsize=11)
# plt.grid(True, linestyle=':', alpha=0.7)
# plt.tight_layout()
# plt.savefig(os.path.join(RESULT_DIR, "attack_primary_rate_crash.png"), dpi=300)

# # 3. Interference Plot
# plt.figure(figsize=(10, 5))
# plt.axhline(y=primary_I_max, color='black', linestyle='-', linewidth=2, label=f'Primary Interference Limit (${{I_{{max}}}}={primary_I_max}$)')
# plt.plot(x_indices, pos_avg_interf_true, label='True Optimal Interference', color='blue', linestyle='--', marker='o', linewidth=2)
# plt.plot(x_indices, pos_avg_interf_pred, label='LLM Agent Interference', color='red', linestyle='-', marker='x', linewidth=2, markersize=8)
# plt.xlabel('Secondary Network Position [x, y]', fontsize=11)
# plt.ylabel('Average Max Interference Injected', fontsize=11)
# plt.xticks(x_indices, position_labels)
# plt.legend(fontsize=11, loc='upper right')
# plt.grid(True, linestyle=':', alpha=0.7)
# plt.tight_layout()
# plt.savefig(os.path.join(RESULT_DIR, "attack_interference_impact.png"), dpi=300)

# # 4. Topology Movement Map
# plt.figure(figsize=(8, 8))
# plt.scatter(80, 80, marker='*', color='gold', s=400, edgecolors='black', zorder=6, label='Primary Tx [80, 80]')

# boundary_radius = 35.0
# circle = plt.Circle((80, 80), boundary_radius, color='gray', fill=False, linestyle='--', linewidth=2, label='Primary Protection Zone')
# plt.gca().add_patch(circle)

# colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
# for idx, (pos, color) in enumerate(zip(positions, colors)):
#     plt.scatter(pos[0], pos[1], color=color, s=150, zorder=5, label=f'Pos {idx+1}: {pos}')

# plt.xlabel('X Coordinate (m)', fontsize=12)
# plt.ylabel('Y Coordinate (m)', fontsize=12)
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
# plt.axis('equal')
# plt.tight_layout()
# plt.savefig(os.path.join(RESULT_DIR, "topology_map.png"), dpi=300, bbox_inches="tight")
# plt.show()