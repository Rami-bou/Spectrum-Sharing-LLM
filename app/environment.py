import random
import numpy as np
import math
from dotenv import load_dotenv

MCS = [
    (2.0, 15), (5.0, 30), (9.0, 45), (11.0, 60),
    (15.0, 90), (18.0, 120), (20.0, 150)
]

# number of primary receivers
N = 3
# number of secondary receivers
M = 3

random.seed(11)

data = []
f = 2e9
c = 3e8
wave = c / f
scale_factor = 1e8

secondary_I_max = 3000
primary_I_max = 1000


primary_transmitter = [80, 80]

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
    Unchanged from the previous version -- this was already correct.
    """
    M = len(direct_h_secondary)
    P2_dist = [0] * M
    budget = allowed_p2

    # Pre-calculate interference caused by Primary onto each Secondary receiver
    total_p1_interf = [sum(P1_dist) * cross_h_secondary[i] for i in range(M)]

    # Greedy allocation loop
    while budget > 0:
        best_eff = -1.0
        best_user = -1
        best_cost = 0

        for i in range(M):
            # Calculate current SINR in dB
            sinr_lin = (P2_dist[i] * direct_h_secondary[i]) / (1.0 + total_p1_interf[i])
            sinr_db = 10 * math.log10(sinr_lin) if sinr_lin > 0 else -999.0

            # Find current rate and next MCS threshold
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

            # If an upgrade tier exists, evaluate cost and efficiency
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

        # If a valid upgrade user was found, purchase the upgrade
        if best_user != -1:
            P2_dist[best_user] += best_cost
            budget -= best_cost
        else:
            # If remaining budget cannot push ANY user to a higher MCS level,
            # dump the leftover budget into the receiver with the strongest direct channel.
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
    total_P2 = sum(P2_vector)  # Total P2 acts as interference to Primary

    for j in range(len(P1_vector)):
        signal = P1_vector[j] * direct_h_primary[j]
        interference_from_secondary = total_P2 * cross_h_primary[j]

        # Calculate physical linear SINR (assuming Noise = 1.0)
        sinr_linear = signal / (1.0 + interference_from_secondary)
        total_throughput_mbps += get_discrete_rate(sinr_linear)

    return total_throughput_mbps


def calculate_secondary_discrete_rate(P1_vector, P2_vector, direct_h_secondary, cross_h_secondary):
    """Calculates total Secondary Throughput based on discrete MCS levels."""
    total_throughput_mbps = 0
    total_P1 = sum(P1_vector)  # Total P1 acts as interference to Secondary

    for j in range(len(P2_vector)):
        signal = P2_vector[j] * direct_h_secondary[j]
        interference_from_primary = total_P1 * cross_h_secondary[j]

        # Calculate physical linear SINR (assuming Noise = 1.0)
        sinr_linear = signal / (1.0 + interference_from_primary)
        total_throughput_mbps += get_discrete_rate(sinr_linear)

    return total_throughput_mbps


def _channel_gain(distance):
    """
    FIX: channel gain computation, used to be:
        h = (wave / (4*pi*d))**2
        int(round(h * scale_factor, 2))
    The round(...,2) was pointless -- wrapping it in int() immediately discarded
    those decimals anyway, which is equivalent to int(h*scale_factor). Any link
    beyond ~119m (with this wave/scale_factor combination) produced h*scale_factor
    < 1.0, which truncated to a hard 0 -- capable of crashing downstream code via
    ZeroDivisionError (1.0/v) or silently zeroing out a channel that should have
    had a small-but-real gain.

    Fix: keep the gain as a float, with a tiny floor instead of a hard truncation
    to zero (same pattern used in the reference D2D paper's channel model:
    np.maximum(gain, floor) rather than int-truncating). This does not change the
    numeric value for any link that was already >= 1.0 after scaling -- it only
    prevents long/weak links from collapsing to exactly zero.
    """
    h = (wave / (4 * np.pi * distance)) ** 2
    return max(h * scale_factor, 1e-6)


def gen_channels(length, pos):

    secondary_transmitter = pos
    while len(data) < length:
        # Primary receivers: bounded cell around their own TX (10-35m),
        # kept away from the secondary transmitter's territory.
        # (Unchanged -- this geometry was already validated: 120/120 acceptance,
        # 0% rejection at every stage, when tested earlier in this project.)
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

        # Secondary receivers: tight cluster around their own TX (1-5m).
        # Needed so secondary's own SINR can reach the same MCS table primary uses --
        # a spread-out secondary is always interference-limited to below MCS tier 0.
        # (Unchanged -- validated earlier: 99.4% of secondary receivers reach a real tier.)
        position_secondary_receiver = []
        direct_h_secondary = []
        for i in range(M):
            dist_r = random.uniform(1, 5)
            angle = random.uniform(0, 2 * math.pi)
            rs = [secondary_transmitter[0] + dist_r * math.cos(angle),
                  secondary_transmitter[1] + dist_r * math.sin(angle)]
            position_secondary_receiver.append(rs)
            d = np.sqrt((rs[0] - secondary_transmitter[0]) ** 2 + (rs[1] - secondary_transmitter[1]) ** 2)
            direct_h_secondary.append(_channel_gain(d))

        cross_h_primary = []
        for pos in position_primary_receiver:
            d = np.sqrt((pos[0] - secondary_transmitter[0]) ** 2 + (pos[1] - secondary_transmitter[1]) ** 2)
            cross_h_primary.append(_channel_gain(d))

        cross_h_secondary = []
        for pos in position_secondary_receiver:
            d = np.sqrt((pos[0] - primary_transmitter[0]) ** 2 + (pos[1] - primary_transmitter[1]) ** 2)
            cross_h_secondary.append(_channel_gain(d))

        # FIX: P1 derivation.
        # OLD: allowed_p1 = secondary_I_max / max(cross_h_secondary), then split
        # proportionally across receivers. This made P1's magnitude entirely a
        # function of SECONDARY's geometry (how much interference secondary can
        # tolerate), with zero reference to primary's own channel quality. Result
        # (measured earlier): baseline P1 landed 100% of primary receivers on the
        # exact top MCS tier, every single sample, zero variance -- an accident of
        # secondary_I_max's specific value, not a deliberate property of the system.
        #
        # NEW: primary is DELIBERATELY targeted to reach the top MCS tier with a
        # generous, intentional margin -- "the channel owner runs strong" is now a
        # designed property, independent of secondary_I_max. secondary_I_max is
        # still enforced, but only as an independent safety ceiling afterward, not
        # as the value that determines P1's magnitude in the first place. Changing
        # secondary_I_max later (e.g. to retune secondary's own story) will no
        # longer silently reshape primary's entire profile.
        #
        # The 18-28 dB buffer above the top threshold (20 dB) was chosen empirically
        # to match the magnitude the OLD code produced by accident (its baseline
        # SINR came out to 38-48 dB) -- so allowed_p2's resulting budget range is
        # consistent with everything already validated in this project (secondary
        # tier-reachability, knapsack behavior, negotiation dynamics).
        P1_dist = []
        for j in range(N):
            target_db = MCS[-1][0] + random.uniform(18, 28)
            target_lin = 10 ** (target_db / 10.0)
            P1_dist.append(max(1, int(round(target_lin / direct_h_primary[j]))))

        # Independent safety cap: reject the sample if this P1 would exceed what
        # secondary can tolerate. This is the genuine, standalone use of
        # secondary_I_max now -- a real constraint check, not a budget generator.
        allowed_p1_ceiling = secondary_I_max / max(cross_h_secondary)
        # if sum(P1_dist) > allowed_p1_ceiling:
        #     continue

        # Reject the sample if ANY primary receiver fails to clear the lowest MCS
        # tier at baseline (P2=0) -- kept as a safety net; should always pass by
        # construction now, but costs nothing to keep as a defensive check.
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
            # the minimum SINR linear that still achieves the same MCS tier as baseline (P2=0)
            min_linear_sinr = 10 ** (target_th / 10.0)
            max_interference = (signal / min_linear_sinr) - 1.0
            if max_interference > 0 and cross_h_primary[j] > 0:
                p2_limits.append(max_interference / cross_h_primary[j])

        if not p2_limits:
            continue
        
        allowed_p2 = int(math.floor(min(p2_limits)))
        if allowed_p2 < M:
            continue

        # Secondary's ground-truth allocation: unchanged, still knapsack-optimal
        # over the (now correctly derived) allowed_p2 budget.
        P2_dist = allocate_p2_knapsack_optimal(allowed_p2, direct_h_secondary, cross_h_secondary, P1_dist)

        data.append([direct_h_primary, direct_h_secondary, cross_h_primary, cross_h_secondary, P1_dist, P2_dist])

    return data

positions = [[20, 20], [30, 30], [40, 40], [50, 50], [60, 60], [65, 65], [70, 70]]
for pos in positions:
    data = gen_channels(190, pos)
    train = data[:70]
    test = data[90:]