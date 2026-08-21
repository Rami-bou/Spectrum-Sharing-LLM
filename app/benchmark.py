# import os
# import csv
# from datetime import datetime
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# from graph import app
# from environment import primary_I_max, calculate_secondary_discrete_rate, gen_channels, MCS, M, train, test, calculate_primary_discrete_rate, get_mcs_threshold

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# RESULT_DIR = os.path.join("results", f"baseline_{timestamp}")
# os.makedirs(RESULT_DIR, exist_ok=True)

# def save_file(metrics_path):
#     with open(metrics_path, "w") as f:
#         f.write("=" * 60 + "\n")
#         f.write("BASELINE PERFORMANCE\n")
#         f.write("=" * 60 + "\n\n")

#         f.write(f"Average Secondary Rate      : {np.mean(se_pred_list):.2f} Mbps\n")
#         f.write(f"Average Optimal Rate        : {np.mean(se_true_list):.2f} Mbps\n")

#         efficiency = 100 * np.mean(se_pred_list) / np.mean(se_true_list)

#         f.write(f"Efficiency                 : {efficiency:.2f} %\n\n")

#         f.write(f"Average Interference        : {np.mean(interf_pred_list):.2f}\n")
#         f.write(f"Maximum Interference        : {np.max(interf_pred_list):.2f}\n")

#         violation_rate = 100 * np.mean(violation_list)

#         f.write(f"Violation Rate             : {violation_rate:.2f} %\n\n")

#         #f.write(f"Average Negotiation Rounds : {np.mean(rounds_list):.2f}\n")

#         success_rate = 100 * np.mean(success_list)

#         f.write(f"Negotiation Success Rate   : {success_rate:.2f} %\n")


# # POISON_FACTOR = 0.1


# # def compute_worst_margin(P1_vector, P2_vector, direct_h_primary, cross_h_primary):
# #     total_p2 = sum(P2_vector)
# #     margins = []
# #     for j in range(len(P1_vector)):
# #         signal = P1_vector[j] * direct_h_primary[j]
# #         if signal <= 0: continue
# #         baseline_sinr_db = 10 * math.log10(signal)
# #         target_th = get_mcs_threshold(baseline_sinr_db)
# #         if target_th < 0: continue
# #         interference = total_p2 * cross_h_primary[j]
# #         actual_sinr_linear = signal / (1.0 + interference)
# #         actual_sinr_db = 10 * math.log10(actual_sinr_linear) if actual_sinr_linear > 0 else -999
# #         margins.append(actual_sinr_db - target_th)
# #     return min(margins) if margins else -999.0

# se_pred_list = []
# se_true_list = []

# interf_pred_list = []
# interf_true_list = []

# rounds_list = []
# success_list = []
# violation_list = []
# se_pred_list_primary = []
# se_true_list_primary = []

# csv_path = os.path.join(RESULT_DIR, "benchmark.csv")
# csv_file = open(csv_path, "w", newline="")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow([
# "Sample", "TrueRate", "PredRate",
# "TruePrimaryRate", "PredPrimaryRate",
# "TrueInterference", "PredInterference", "Violation", "Rounds", "Decision", "TrueP2", "PredP2"
# ])

# print(f"\nStarting Benchmark over {len(test)} Test Samples...")

# for i in range(len(test)):
#     direct_h_sec = test[i][1]
#     cross_h_sec = test[i][3]
#     direct_h_prim = test[i][0]
#     cross_h_prim = test[i][2]
#     true_p1 = test[i][4]
#     true_p2 = test[i][5]

#     # poisoned_cross_h_pri = [max(1, v * POISON_FACTOR) for v in cross_h_prim]

#     initial_state = {
#     "direct_primary_channels": test[i][0],
#     "direct_secondary_channels": test[i][1],
#     "cross_primary_channels": test[i][2],
#     "cross_secondary_channels": test[i][3],
#     "P1": test[i][4],
#     "P2": [0] * M,
#     "primary_critique": "",
#     "secondary_critique": "",
#     "primary_decision": "",
#     "delta_hist": [],
#     "iteration": 0
#     }

#     result = app.invoke(initial_state)

#     pred_p2 = result['P2']

#     # margin_believed = compute_worst_margin(true_p1, pred_p2, direct_h_prim, poisoned_cross_h_pri)
#     # margin_actual = compute_worst_margin(true_p1, pred_p2, direct_h_prim, cross_h_prim)

#     # 1. Calculate Discrete Secondary Rates
#     rate_pred = calculate_secondary_discrete_rate(true_p1, pred_p2, direct_h_sec, cross_h_sec)
#     rate_true = calculate_secondary_discrete_rate(true_p1, true_p2, direct_h_sec, cross_h_sec)
#     se_pred_list.append(rate_pred)
#     se_true_list.append(rate_true)

#     rate_pred_primary = calculate_primary_discrete_rate(true_p1, pred_p2, direct_h_prim, cross_h_prim)
#     rate_true_primary = calculate_primary_discrete_rate(true_p1, true_p2, direct_h_prim, cross_h_prim)
#     se_pred_list_primary.append(rate_pred_primary)
#     se_true_list_primary.append(rate_true_primary)


#     # 2. Calculate Worst-Case Caused Interference on Primary Receivers

#     max_interf_pred = sum(pred_p2) * max(cross_h_prim)
#     max_interf_true = sum(true_p2) * max(cross_h_prim)
#     interf_pred_list.append(max_interf_pred)
#     interf_true_list.append(max_interf_true)
#     # 3. Calculate the sucess rate
#     success_list.append(1 if result["primary_decision"] == "ACCEPT" else 0)
#     # 4. Constraint violation
#     violation_list.append(1 if max_interf_pred > primary_I_max else 0)

#     csv_writer.writerow([
#     i + 1, rate_true, rate_pred,
#     rate_true_primary, rate_pred_primary,
#     max_interf_true, max_interf_pred, violation_list[i],
#     result["iteration"], result["primary_decision"], sum(true_p2), sum(pred_p2)
#     ])

#     print(f"Sample {i+1}/100 | True Rate: {rate_true} | Pred Rate: {rate_pred} | Pred Interf: {max_interf_pred:.1f}")
#     print(f"True P2: {true_p2}")
#     print(f"pred P2: {result['P2']}")


# csv_file.close()

# metrics_path = os.path.join(RESULT_DIR, "metrics.txt")

# save_file(metrics_path)

# print(f"System Benchmark Before attack:\n")
# print(f"Average Secondary Rate (True): {np.mean(se_true_list):.2f}")
# print(f"Average Secondary Rate (Predicted): {np.mean(se_pred_list):.2f}")
# print(f"Average Interference (Predicted): {np.mean(interf_pred_list):.2f}")
# print(f"Max Interference (Predicted): {np.max(interf_pred_list):.2f}")
# print(f"Efficiency: {np.mean(success_list):.0%}")
# print(f"Constraint Violations: {np.sum(violation_list):.0%}")

# # Only attempt to plot if there are enough test samples for binning
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

import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from graph import app
from environment import primary_I_max, calculate_secondary_discrete_rate, gen_channels, MCS, M, calculate_primary_discrete_rate, get_mcs_threshold

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join("results", f"spatial_benchmark_{timestamp}")
os.makedirs(RESULT_DIR, exist_ok=True)

# 1. Define the positions you want to test
positions = [[20, 20], [30, 30], [40, 40], [50, 50], [60, 60], [65, 65], [70, 70]]
position_labels = [f"[{x},{y}]" for x, y in positions]

# We will store the AVERAGE results for each position here
pos_avg_se_pred = []
pos_avg_se_true = []
pos_avg_prim_pred = []
pos_avg_prim_true = []
pos_avg_interf_pred = []
pos_avg_interf_true = []

print(f"Starting Spatial Benchmark over {len(positions)} positions...\n")

# =====================================================================
# SIMULATION LOOP (Iterating over positions)
# =====================================================================
for pos_idx, pos in enumerate(positions):
    print(f"{'='*50}")
    print(f"Evaluating Secondary Position: {pos}")
    print(f"{'='*50}")
    
    # [CRITICAL STEP]: Generate new test data for this specific position
    # You must adapt the arguments below to match your environment.py definition!
    # e.g., test_samples = gen_channels(num_samples=100, sec_position=pos)
    test_samples = gen_channels(sec_pos=pos) 
    
    se_pred_list = []
    se_true_list = []
    se_pred_list_primary = []
    se_true_list_primary = []
    interf_pred_list = []
    interf_true_list = []
    violation_list = []
    
    for i in range(len(test_samples)):
        direct_h_prim = test_samples[i][0]
        direct_h_sec = test_samples[i][1]
        cross_h_prim = test_samples[i][2]
        cross_h_sec = test_samples[i][3]
        true_p1 = test_samples[i][4]
        true_p2 = test_samples[i][5]

        initial_state = {
            "direct_primary_channels": direct_h_prim,
            "direct_secondary_channels": direct_h_sec,
            "cross_primary_channels": cross_h_prim,
            "cross_secondary_channels": cross_h_sec,
            "P1": true_p1,
            "P2": [0] * M,
            "primary_critique": "",
            "secondary_critique": "",
            "primary_decision": "",
            "delta_hist": [],
            "iteration": 0
        }

        # Run LLM Agent
        result = app.invoke(initial_state)
        pred_p2 = result['P2']

        # Calculate Rates
        rate_pred = calculate_secondary_discrete_rate(true_p1, pred_p2, direct_h_sec, cross_h_sec)
        rate_true = calculate_secondary_discrete_rate(true_p1, true_p2, direct_h_sec, cross_h_sec)
        se_pred_list.append(rate_pred)
        se_true_list.append(rate_true)

        rate_pred_primary = calculate_primary_discrete_rate(true_p1, pred_p2, direct_h_prim, cross_h_prim)
        rate_true_primary = calculate_primary_discrete_rate(true_p1, true_p2, direct_h_prim, cross_h_prim)
        se_pred_list_primary.append(rate_pred_primary)
        se_true_list_primary.append(rate_true_primary)

        # Calculate Interference
        max_interf_pred = sum(pred_p2) * max(cross_h_prim)
        max_interf_true = sum(true_p2) * max(cross_h_prim)
        interf_pred_list.append(max_interf_pred)
        interf_true_list.append(max_interf_true)
        
        print(f"  Sample {i+1} | Pred Sec Rate: {rate_pred:.1f} | Pred Interf: {max_interf_pred:.1f}")

    # Store the averages for this position
    pos_avg_se_pred.append(np.mean(se_pred_list))
    pos_avg_se_true.append(np.mean(se_true_list))
    
    pos_avg_prim_pred.append(np.mean(se_pred_list_primary))
    pos_avg_prim_true.append(np.mean(se_true_list_primary))
    
    pos_avg_interf_pred.append(np.mean(interf_pred_list))
    pos_avg_interf_true.append(np.mean(interf_true_list))


# =====================================================================
# PLOTTING METRICS VS POSITIONS
# =====================================================================
x_indices = np.arange(len(positions)) # 0, 1, 2... for the x-axis

# 1. Secondary Rate vs Position
plt.figure(figsize=(10, 5))
plt.plot(x_indices, pos_avg_se_true, label='True Optimal Secondary Rate', color='blue', linestyle='--', marker='o', linewidth=2)
plt.plot(x_indices, pos_avg_se_pred, label='LLM Agent Secondary Rate', color='red', linestyle='-', marker='s', linewidth=2)
plt.title('Secondary Network Sum Rate by Spatial Position', fontsize=13)
plt.xlabel('Secondary Network Position [x, y]', fontsize=11)
plt.ylabel('Average Secondary Rate (Mbps)', fontsize=11)
plt.xticks(x_indices, position_labels)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "secondary_rate_spatial.png"), dpi=300)

# 2. Primary Rate vs Position
plt.figure(figsize=(10, 5))
plt.plot(x_indices, pos_avg_prim_true, label='True Optimal Primary Rate', color='blue', linestyle='--', marker='o', linewidth=2)
plt.plot(x_indices, pos_avg_prim_pred, label='LLM Agent Primary Rate', color='red', linestyle='-', marker='s', linewidth=2)
plt.title('Primary Network Sum Rate by Spatial Position', fontsize=13)
plt.xlabel('Secondary Network Position [x, y]', fontsize=11)
plt.ylabel('Average Primary Rate (Mbps)', fontsize=11)
plt.xticks(x_indices, position_labels)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "primary_rate_spatial.png"), dpi=300)

# 3. Interference vs Position
plt.figure(figsize=(10, 5))
plt.axhline(y=primary_I_max, color='black', linestyle='-', linewidth=2, label=f'Primary Interference Limit (${{I_{{max}}}}={primary_I_max}$)')
plt.plot(x_indices, pos_avg_interf_true, label='True Optimal Interference', color='blue', linestyle='--', marker='o', linewidth=2)
plt.plot(x_indices, pos_avg_interf_pred, label='LLM Agent Interference', color='red', linestyle='-', marker='x', linewidth=2, markersize=8)
plt.title('Primary Network Protection: Injected Interference by Position', fontsize=13)
plt.xlabel('Secondary Network Position [x, y]', fontsize=11)
plt.ylabel('Average Max Interference Injected', fontsize=11)
plt.xticks(x_indices, position_labels)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "interference_spatial.png"), dpi=300)


# =====================================================================
# TOPOLOGY VISUALIZATION (Plotting the movement map)
# =====================================================================
plt.figure(figsize=(8, 8))

# Assume Primary Tx is at origin (0,0)
plt.scatter(0, 0, marker='*', color='gold', s=400, edgecolors='black', label='Primary Tx (Origin)')

# Draw a circle representing the primary boundary (radius calculated to just touch [65,65])
# Distance to [65,65] is sqrt(65^2 + 65^2) = 91.92
boundary_radius = math.sqrt(65**2 + 65**2)
circle = plt.Circle((0, 0), boundary_radius, color='gray', fill=False, linestyle='--', linewidth=2, label=f'Primary Boundary (R$\approx${boundary_radius:.1f})')
plt.gca().add_patch(circle)

# Plot all secondary positions
colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
for idx, (pos, color) in enumerate(zip(positions, colors)):
    plt.scatter(pos[0], pos[1], color=color, s=150, zorder=5, label=f'Sec Pos {idx+1}: {pos}')

plt.title('Spatial Topology: Secondary Network Movement', fontsize=14)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10) # Place legend outside
plt.axis('equal') # Keep aspect ratio square so the circle isn't distorted
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "topology_map.png"), dpi=300, bbox_inches="tight")

plt.show()