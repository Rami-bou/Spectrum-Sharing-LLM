import random
import numpy as np
import math

MCS = [
    (2.0, 15), (5.0, 30), (9.0, 45), (11.0, 60),
    (15.0, 90), (18.0, 120), (20.0, 150)
]

# number of primary receivers
N = 3
# number of secondary receivers
M = 3

random.seed(11)

f = 2e9
c = 3e8
wave = c / f
scale_factor = 1e8

secondary_I_max = 3000
primary_I_max = 1000

primary_transmitter = [80, 80]
# Default position, but can now be overridden
secondary_transmitter_default = [70, 70]


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
    Distributes allowed_p2 among secondary receivers to maximize aggregate
    discrete throughput using greedy Knapsack selection.
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
                    value = next_rate - curr_rate
                    eff = value / float(cost)

                    if eff > best_eff:
                        best_eff = eff
                        best_user = i
                        best_cost = cost

        if best_user != -1:
            P2_dist[best_user] += best_cost
            budget -= best_cost
        else:
            best_user = max(range(M), key=lambda k: direct_h_secondary[k])
            P2_dist[best_user] += budget
            budget = 0

    return P2_dist


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
        sinr_linear = signal / (1.0 + interference_from_secondary)
        total_throughput_mbps += get_discrete_rate(sinr_linear)

    return total_throughput_mbps


def calculate_secondary_discrete_rate(P1_vector, P2_vector, direct_h_secondary, cross_h_secondary):
    """Calculates total Secondary Throughput based on discrete MCS levels."""
    total_throughput_mbps = 0
    total_P1 = sum(P1_vector) 

    for j in range(len(P2_vector)):
        signal = P2_vector[j] * direct_h_secondary[j]
        interference_from_primary = total_P1 * cross_h_secondary[j]
        sinr_linear = signal / (1.0 + interference_from_primary)
        total_throughput_mbps += get_discrete_rate(sinr_linear)

    return total_throughput_mbps


def _channel_gain(distance):
    h = (wave / (4 * np.pi * distance)) ** 2
    return max(h * scale_factor, 1e-6)


def gen_channels(length, sec_pos=None):
    """
    Generates channel samples.
    :param length: Number of samples to generate.
    :param sec_pos: The [x, y] coordinates of the secondary transmitter.
    """
    # 1. Initialize data list locally so we generate fresh data every time
    local_data = []
    
    # 2. Use dynamically passed position, or fallback to default
    sec_tx = sec_pos if sec_pos is not None else secondary_transmitter_default

    while len(local_data) < length:
        position_primary_receiver = []
        direct_h_primary = []
        for i in range(N):
            dist_r = random.uniform(10, 35)
            angle = random.uniform(0, 2 * math.pi)
            rp = [primary_transmitter[0] + dist_r * math.cos(angle),
                  primary_transmitter[1] + dist_r * math.sin(angle)]
            position_primary_receiver.append(rp)
            d = np.sqrt((rp[0] - primary_transmitter[0]) ** 2 + (rp[1] - primary_transmitter[1]) ** 2)
            direct_h_primary.append(_channel_gain(d))

        position_secondary_receiver = []
        direct_h_secondary = []
        for i in range(M):
            dist_r = random.uniform(1, 5)
            angle = random.uniform(0, 2 * math.pi)
            rs = [sec_tx[0] + dist_r * math.cos(angle),
                  sec_tx[1] + dist_r * math.sin(angle)]
            position_secondary_receiver.append(rs)
            d = np.sqrt((rs[0] - sec_tx[0]) ** 2 + (rs[1] - sec_tx[1]) ** 2)
            direct_h_secondary.append(_channel_gain(d))

        cross_h_primary = []
        for pos in position_primary_receiver:
            d = np.sqrt((pos[0] - sec_tx[0]) ** 2 + (pos[1] - sec_tx[1]) ** 2)
            cross_h_primary.append(_channel_gain(d))

        cross_h_secondary = []
        for pos in position_secondary_receiver:
            d = np.sqrt((pos[0] - primary_transmitter[0]) ** 2 + (pos[1] - primary_transmitter[1]) ** 2)
            cross_h_secondary.append(_channel_gain(d))

        P1_dist = []
        for j in range(N):
            target_db = MCS[-1][0] + random.uniform(18, 28)
            target_lin = 10 ** (target_db / 10.0)
            P1_dist.append(max(1, int(round(target_lin / direct_h_primary[j]))))

        baseline_ok = True
        for j in range(N):
            signal = P1_dist[j] * direct_h_primary[j]
            if signal <= 0 or get_mcs_threshold(10 * math.log10(signal)) < 0:
                baseline_ok = False
                break
        if not baseline_ok:
            continue

        p2_limits = []
        for j in range(N):
            signal = P1_dist[j] * direct_h_primary[j]
            baseline_sinr_db = 10 * math.log10(signal)
            target_th = get_mcs_threshold(baseline_sinr_db)
            min_linear_sinr = 10 ** (target_th / 10.0)
            max_interference = (signal / min_linear_sinr) - 1.0
            if max_interference > 0 and cross_h_primary[j] > 0:
                p2_limits.append(max_interference / cross_h_primary[j])

        if not p2_limits:
            continue
        
        allowed_p2 = int(math.floor(min(p2_limits)))
        if allowed_p2 < M:
            continue

        P2_dist = allocate_p2_knapsack_optimal(allowed_p2, direct_h_secondary, cross_h_secondary, P1_dist)

        local_data.append([direct_h_primary, direct_h_secondary, cross_h_primary, cross_h_secondary, P1_dist, P2_dist])

    return local_data


# If you need default test data initialized on import (optional, for backwards compatibility)
data = gen_channels(190)
train = data[:70]
test = data[90:]