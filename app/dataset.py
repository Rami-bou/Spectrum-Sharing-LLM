import random
import numpy as np
import math
from dotenv import load_dotenv

N = 5  
M = 5 

data = []

f = 2e9
c = 3e8
wave = c / f
scale_factor = 1e8

secondary_I_max = 8000
random.seed(10)

MCS = [
    (2.0, 15),
    (5.0, 30),
    (9.0, 45),
    (11.0, 60),
    (15.0, 90),
    (18.0, 120),
    (20.0, 150)
]

def get_mcs_threshold(sinr_db):
    """Finds the minimum required SINR (dB) for the current state."""
    target_th = -999
    for th, rate in MCS:
        if sinr_db >= th:
            target_th = th
        else:
            break
    return target_th

def allocate_p2_knapsack_optimal(allowed_p2, direct_h_secondary, cross_h_secondary, P1_dist):
    """
    Distributes allowed_p2 to maximize Proportional Fairness (Sum of Log Rates)
    using greedy Knapsack selection.
    """
    M = len(direct_h_secondary)
    P2_dist = [0] * M
    budget = allowed_p2

    total_p1_interf = [sum(P1_dist) * cross_h_secondary[i] for i in range(M)]

    while budget > 0:
        best_eff = -1.0
        best_user = -1
        best_cost = 0

        for i in range(M):
            # Calculate current SINR in dB
            sinr_lin = (P2_dist[i] * direct_h_secondary[i]) / (1.0 + total_p1_interf[i])
            sinr_db = 10 * math.log10(sinr_lin) if sinr_lin > 0 else -999.0

            curr_rate = 0
            next_th = None
            next_rate = 0

            for th, rate in MCS:
                if sinr_db >= th:
                    curr_rate = rate
                elif next_th is None:
                    next_th = th
                    next_rate = rate
                    break

            if next_th is not None:
                target_sinr_lin = 10 ** (next_th / 10.0)
                required_p2 = (target_sinr_lin * (1.0 + total_p1_interf[i])) / direct_h_secondary[i]
                cost = int(math.ceil(required_p2 - P2_dist[i]))

                if 0 < cost <= budget:
                    # PROPORTIONAL FAIRNESS METRIC: Logarithmic Utility Gain
                    log_val_curr = math.log(1.0 + curr_rate)
                    log_val_next = math.log(1.0 + next_rate)
                    value = log_val_next - log_val_curr

                    eff = value / float(cost)

                    if eff > best_eff:
                        best_eff = eff
                        best_user = i
                        best_cost = cost

        if best_user != -1:
            P2_dist[best_user] += best_cost
            budget -= best_cost
        else:
            # If leftover budget can't buy any discrete step, assign to best channel
            best_user = max(range(M), key=lambda k: direct_h_secondary[k])
            P2_dist[best_user] += budget
            budget = 0

    return P2_dist

def gen_channels(length):
    while len(data) < length:
        primary_transmitter = [8, 35]
        secondary_transmitter = [5, -20]

        # 1. Primary Channels
        dist__state = random.randint(1, 3)
        position_primary_receiver = []
        direct_h_primary = []
        for i in range(N):
            if dist__state == 1:
                rp = [random.uniform(10, 15), random.uniform(30, 50)]
            elif dist__state == 2:
                rp = [random.uniform(16, 25), random.uniform(30, 50)]
            else:
                rp = [random.uniform(25, 50), random.uniform(30, 50)]

            position_primary_receiver.append(rp)
            d = np.sqrt((rp[0]-primary_transmitter[0])**2 + (rp[1]-primary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            direct_h_primary.append(int(round(h * scale_factor, 2)))

        # 2. Secondary Channels
        dist__state = random.randint(1, 3)
        position_secondary_receiver = []
        direct_h_secondary = []
        for i in range(M):
            if dist__state == 1:
                rs = [random.uniform(10, 15), random.uniform(0, 29)]
            elif dist__state == 2:
                rs = [random.uniform(16, 25), random.uniform(0, 29)]
            else:
                rs = [random.uniform(25, 50), random.uniform(0, 29)]

            position_secondary_receiver.append(rs)
            d = np.sqrt((rs[0]-secondary_transmitter[0])**2 + (rs[1]-secondary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            direct_h_secondary.append(int(round(h * scale_factor, 2)))

        # 3. Cross Channels
        cross_h_primary = []
        for pos in position_primary_receiver:
            d = np.sqrt((pos[0]-secondary_transmitter[0])**2 + (pos[1]-secondary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            cross_h_primary.append(int(round(h * scale_factor, 2)))

        cross_h_secondary = []
        for pos in position_secondary_receiver:
            d = np.sqrt((pos[0]-primary_transmitter[0])**2 + (pos[1]-primary_transmitter[1])**2)
            h = (wave / (4 * np.pi * d))**2
            cross_h_secondary.append(int(round(h * scale_factor, 2)))

        # 4. P1 Power Distribution
        allowed_p1 = int(round(secondary_I_max / max(cross_h_secondary)))
        if allowed_p1 < N:
            continue

        inverses = [1.0 / v for v in direct_h_primary]
        sum_inverses = sum(inverses)
        P1_dist = [int(round((inv / sum_inverses) * allowed_p1)) for inv in inverses]
        # 5. Calculate Max Allowed P2 based on Primary MCS Cliffs
        p2_limits = []
        for j in range(N):
            signal = P1_dist[j] * direct_h_primary[j]
            if signal <= 0:
                continue
            
            # Baseline SINR in dB (when P2 = 0)
            baseline_sinr_db = 10 * math.log10(signal)
            
            # Target MCS cliff threshold
            target_th = get_mcs_threshold(baseline_sinr_db)
            if target_th < 0:  # Skip if user can't even reach MCS 0
                continue
            
            # Max allowed linear interference before dropping below target_th
            min_linear_sinr = 10 ** (target_th / 10.0)
            max_interference = (signal / min_linear_sinr) - 1.0
            
            if max_interference > 0 and cross_h_primary[j] > 0:
                p2_limits.append(max_interference / cross_h_primary[j])

        if not p2_limits:
            continue

        allowed_p2 = int(math.floor(min(p2_limits)))
        if allowed_p2 < M:
            continue

        # 6. P2 Power Distribution Knapsack Optimization
        P2_dist = allocate_p2_knapsack_optimal(allowed_p2, direct_h_secondary, cross_h_secondary, P1_dist)

        data.append([direct_h_primary, direct_h_secondary, cross_h_primary, cross_h_secondary, P1_dist, P2_dist])

    return data
