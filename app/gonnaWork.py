import itertools
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from llama_cpp import Llama

np.set_printoptions(precision=3)


## Check the feasibility of generated location
def Feasible_Loc_Init(Cur_loc, Size_area, Dist_TX_RX):
    temp_dist = 2 * Dist_TX_RX * (np.random.rand(2) - 0.5)
    temp_chan = Cur_loc + temp_dist
    while (np.max(np.abs(temp_chan)) > Size_area / 2) or (
        np.linalg.norm(temp_dist) > Dist_TX_RX
    ):
        temp_dist = 2 * Dist_TX_RX * (np.random.rand(2) - 0.5)
        temp_chan = Cur_loc + temp_dist
    return temp_chan


## Generate location of D2D users and CUE
def loc_init(Size_area, Dist_TX_RX, Num_D2D, Num_Ch):
    tx_loc = Size_area * (np.random.rand(Num_D2D, 2) - 0.5)
    rx_loc = np.zeros((Num_D2D + 1, 2))  # Last index is Cellular Base Station (BS)
    for i in range(Num_D2D):
        rx_loc[i, :] = Feasible_Loc_Init(tx_loc[i, :], Size_area, Dist_TX_RX)

    # BS is located at origin (0,0)
    rx_loc[Num_D2D, :] = np.array([0.0, 0.0])
    tx_loc_CUE = Size_area * (np.random.rand(Num_Ch, 2) - 0.5)

    return rx_loc, tx_loc, tx_loc_CUE


## Generate sample data for channel
def ch_gen(
    Size_area,
    D2D_dist,
    Num_D2D,
    Num_Ch,
    Num_samples,
    PL_alpha=3.8,
    PL_const=34.5,
):
    ch_w_fading = []
    rx_loc_mat = []
    tx_loc_mat = []
    CUE_loc_mat = []

    for i in range(Num_samples):
        rx_loc, tx_loc, tx_loc_CUE = loc_init(
            Size_area, D2D_dist, Num_D2D, Num_Ch
        )
        ch_w_temp_band = []

        for j in range(Num_Ch):
            # Combine D2D Transmitters and 1 Cellular User Equipment Transmitter
            tx_loc_with_CUE = np.vstack((tx_loc, tx_loc_CUE[j : j + 1, :]))

            # Matrix distance calculation: Row = RX, Column = TX
            # dist_vec[r, t] = distance from TX t to RX r
            dist_vec = np.zeros((Num_D2D + 1, Num_D2D + 1))
            for r in range(Num_D2D + 1):
                for t in range(Num_D2D + 1):
                    dist_vec[r, t] = np.linalg.norm(
                        rx_loc[r, :] - tx_loc_with_CUE[t, :]
                    )

            dist_vec = np.maximum(dist_vec, 3.0)  # Safe guard minimum distance

            # Path loss calculation (dB and Linear scale)
            pu_ch_gain_db = -PL_const - 10 * PL_alpha * np.log10(dist_vec)
            pu_ch_gain = 10 ** (pu_ch_gain_db / 10.0)

            # Rayleigh Fading Channel
            multi_fading = 0.5 * (
                np.random.randn(Num_D2D + 1, Num_D2D + 1) ** 2
                + np.random.randn(Num_D2D + 1, Num_D2D + 1) ** 2
            )

            final_ch = np.maximum(pu_ch_gain * multi_fading, np.exp(-30))
            ch_w_temp_band.append(final_ch)

        ch_w_fading.append(ch_w_temp_band)
        rx_loc_mat.append(rx_loc)
        tx_loc_mat.append(tx_loc)
        CUE_loc_mat.append(tx_loc_CUE)

    return (
        np.array(ch_w_fading),
        np.array(rx_loc_mat),
        np.array(tx_loc_mat),
        np.array(CUE_loc_mat),
    )


## Calculate data rate for single channel, single sample
def cal_RATE_one_sample_one_channel(channel, tx_power, noise):
    # channel: (Num_RX, Num_TX), tx_power: (Num_Candidates, Num_TX)
    diag_ch = np.diag(channel)
    inter_ch = channel - np.diag(diag_ch)

    # Signal power received: P_t * H_diag
    sig_power = tx_power * diag_ch

    # Interference power received: Sum_j(P_j * H_ij) for j != i
    int_power = np.matmul(tx_power, inter_ch.T)

    SINR_val = sig_power / (int_power + noise)
    cap_val = np.log2(1.0 + SINR_val)
    return cap_val


def cal_CUE_INTER_one_sample_one_channel(channel, tx_power):
    # Interference caused to CUE RX (index = Num_D2D) by D2D TXs
    num_D2D = channel.shape[0] - 1
    cue_inter_ch = channel[num_D2D, :num_D2D]
    int_power = np.matmul(tx_power[:, :num_D2D], cue_inter_ch)
    return int_power


def cal_rate_NP(channel, tx_power_in, tx_max, noise, DUE_thr, I_thr, P_c):
    num_sample = channel.shape[0]
    num_channel = channel.shape[1]
    num_D2D_user = channel.shape[2] - 1

    tot_SE = 0
    tot_EE = 0
    DUE_violation = 0
    CUE_violation = 0

    tx_power = np.hstack(
        (tx_power_in, np.zeros((tx_power_in.shape[0], 1, num_channel)))
    )

    for i in range(num_sample):
        cur_cap = 0
        DUE_mask = 1
        CUE_mask = 1

        for j in range(num_channel):
            cur_ch = channel[i][j]
            cur_power = tx_power[i, :, j]
            cur_power = np.array([cur_power])
            cur_ch_cap = cal_RATE_one_sample_one_channel(
                cur_ch, cur_power, noise
            )
            inter = cal_CUE_INTER_one_sample_one_channel(cur_ch, cur_power)

            cur_cap = cur_cap + cur_ch_cap[0]
            CUE_mask = CUE_mask * (inter[0] <= I_thr)

        for j in range(num_D2D_user):
            DUE_mask = DUE_mask * (cur_cap[j] >= DUE_thr)

        D2D_SE_sum = np.sum(cur_cap[:-1]) * CUE_mask * DUE_mask
        p_sum = np.sum(tx_power_in[i], axis=0)
        D2D_EE_sum = (
            np.sum(cur_cap[:-1]) / (np.sum(p_sum) + P_c)
        ) * CUE_mask * DUE_mask

        if CUE_mask == 0:
            CUE_violation += 1
        if DUE_mask == 0:
            DUE_violation += 1

        tot_SE += D2D_SE_sum
        tot_EE += D2D_EE_sum

    tot_SE = tot_SE / num_D2D_user / num_sample
    tot_EE = tot_EE / num_D2D_user / num_sample
    PRO_DUE_vio = DUE_violation / num_sample
    PRO_CUE_vio = CUE_violation / num_sample

    return tot_SE, tot_EE, PRO_CUE_vio, PRO_DUE_vio


def all_possible_tx_power(num_channel, num_user, granuty):
    items = [np.linspace(0, 1, granuty)] * (num_user * num_channel)
    temp_power = np.array(list(itertools.product(*items)))
    power = np.reshape(temp_power, (-1, num_user, num_channel))

    power_mat = []
    for i in range(power.shape[0]):
        power_mat.append(power[i])

    return np.array(power_mat)


def optimal_power_w_chan(
    channel, tx_max, noise, DUE_thr, I_thr, P_c, tx_power_set, opt="SE"
):
    num_channel = channel.shape[1]
    num_D2D_user = channel.shape[2] - 1
    num_samples = channel.shape[0]

    power_mat_SE = []
    chan_infea_mat = []

    for i in range(num_samples):
        cur_cap = np.zeros((tx_power_set.shape[0], num_D2D_user + 1))
        DUE_mask = np.ones(tx_power_set.shape[0])
        CUE_mask = np.ones(tx_power_set.shape[0])

        tx_power = tx_max * np.hstack(
            (
                tx_power_set[:, :, 0],
                np.zeros((tx_power_set.shape[0], 1)),
            )
        )

        for j in range(num_channel):
            cur_ch = channel[i][j]
            cur_ch_cap = cal_RATE_one_sample_one_channel(
                cur_ch, tx_power, noise
            )
            inter = cal_CUE_INTER_one_sample_one_channel(cur_ch, tx_power)

            cur_cap += cur_ch_cap
            CUE_mask = CUE_mask * (inter <= I_thr)

        for j in range(num_D2D_user):
            DUE_mask = DUE_mask * (cur_cap[:, j] >= DUE_thr)

        sum_D2D_SE = np.sum(cur_cap[:, :-1], axis=1)
        sum_D2D_EE = sum_D2D_SE / (
            np.sum(tx_power[:, :-1], axis=1) + P_c
        )

        # Apply feasibility penalization
        feasible_mask = CUE_mask * DUE_mask
        sum_D2D_SE_masked = np.where(feasible_mask, sum_D2D_SE, -np.inf)
        sum_D2D_EE_masked = np.where(feasible_mask, sum_D2D_EE, -np.inf)

        if opt == "SE":
            arg_max_val = np.argmax(sum_D2D_SE_masked)
            best_val = sum_D2D_SE_masked[arg_max_val]
        else:
            arg_max_val = np.argmax(sum_D2D_EE_masked)
            best_val = sum_D2D_EE_masked[arg_max_val]

        if best_val == -np.inf:
            found_tx_val = np.zeros((num_D2D_user, 1))
            chan_infea_mat.append(channel[i])
        else:
            found_tx_val = tx_power[arg_max_val, :-1].reshape(num_D2D_user, 1)

        power_mat_SE.append(found_tx_val)

    power_mat_SE = np.array(power_mat_SE)
    tot_SE, tot_EE, PRO_CUE_vio, PRO_DUE_vio = cal_rate_NP(
        channel, power_mat_SE, tx_max, noise, DUE_thr, I_thr, P_c
    )

    return (
        tot_SE,
        tot_EE,
        PRO_CUE_vio,
        PRO_DUE_vio,
        np.array(chan_infea_mat),
        power_mat_SE,
        channel,
    )


def cal_SE_EE(channel, tx_max, noise, DUE_thr, I_thr, P_c, tx_power_mat, opt="SE"):
    num_D2D_user = channel.shape[0] - 1

    tx_power = np.vstack((tx_power_mat, np.zeros((1, 1))))
    tx_power = tx_power.T  # Shape (1, Num_TX)

    cur_ch_cap = cal_RATE_one_sample_one_channel(channel, tx_power, noise)
    inter = cal_CUE_INTER_one_sample_one_channel(channel, tx_power)

    cue_ok = inter[0] <= I_thr
    due_ok = np.all(cur_ch_cap[0, :-1] >= DUE_thr)

    if cue_ok and due_ok:
        sum_D2D_SE = np.sum(cur_ch_cap[0, :-1])
        sum_D2D_EE = sum_D2D_SE / (np.sum(tx_power_mat) + P_c)
    else:
        sum_D2D_SE = 0.0
        sum_D2D_EE = 0.0

    return sum_D2D_SE, sum_D2D_EE


# ---------------------------------------------------------
# Simulation Setup & Parameters
# ---------------------------------------------------------
np.random.seed(0)

Num_user = 2
Num_channel = 1
Num_power_level = 11  # Grid resolution (0, 10, 20, ..., 100)
tx_power_set = all_possible_tx_power(Num_channel, Num_user, Num_power_level)

Size_area = 20.0
D2D_dist = 15.0
tx_max = 100.0

DUE_thr = 0.5
I_thr = 10 ** (-55.0 / 10)
P_c = 2 * 10**2.0
BW = 1e7
noise = BW * 10**-17.4

# Initialize Channel Dataset Statistics
Num_sample_stats = 5000
ch_mat_stats, _, _, _ = ch_gen(
    Size_area, D2D_dist, Num_user, Num_channel, Num_sample_stats
)
ch_mat_log = np.log(ch_mat_stats)
chan_avg = np.mean(ch_mat_log)
chan_std = np.std(ch_mat_log)

# Initialize Model
llm = Llama(
    model_path="./models/llama-2-13b.Q5_K_M.gguf",
    n_ctx=4096,
    verbose=False,
)

# Benchmark Validation
critera = "EE"
P_c = 5 * 10**2.0
Size_area = 70.0
D2D_dist = 20.0

ch_mat_val, rx_mat_val, tx_mat_val, CUE_mat_val = ch_gen(
    Size_area, D2D_dist, Num_user, Num_channel, 501
)
(
    SE_OPT_val,
    EE_OPT_val,
    CUE_vio_OPT_val,
    DUE_vio_OPT,
    INF_CHAN_MAT_val,
    PW_VEC_val,
    CHAN_VEC_val,
) = optimal_power_w_chan(
    ch_mat_val,
    tx_max,
    noise,
    DUE_thr,
    I_thr,
    P_c,
    tx_power_set,
    opt=critera,
)

print("Starting LLM-aided Power Control evaluation...\n")

batch_size = 50  # Balanced context prompt size

SE_opt_mat = 0
EE_opt_mat = 0
SE_prop_mat = 0
EE_prop_mat = 0
SE_prop_2_mat = 0
EE_prop_2_mat = 0
SE_rand_mat = 0
EE_rand_mat = 0
SE_bin_mat = 0
EE_bin_mat = 0

for j in range(100):
    ch_mat, rx_mat, tx_mat, CUE_mat = ch_gen(
        Size_area, D2D_dist, Num_user, Num_channel, batch_size
    )
    (
        SE_OPT,
        EE_OPT,
        CUE_vio_OPT,
        DUE_vio_OPT,
        INF_CHAN_MAT,
        PW_VEC,
        CHAN_VEC,
    ) = optimal_power_w_chan(
        ch_mat, tx_max, noise, DUE_thr, I_thr, P_c, tx_power_set, opt=critera
    )

    query_text = (
        "Take a deep breath and work on this problem step-by-step. You are a mathematical tool to predict optimal power levels."
        " Your job is to predict B for given A. The following is the dataset:\n"
    )

    for i in range(batch_size):
        chan_revised = (
            (np.log(ch_mat[i, 0, :Num_user, :Num_user]) - chan_avg)
            / chan_std
            * 100
        )
        query_text += f"If A is {chan_revised[0, 0]:0.0f}, {chan_revised[0, 1]:0.0f}, {chan_revised[1, 0]:0.0f}, {chan_revised[1, 1]:0.0f}, then B is {PW_VEC[i, 0, 0]:0.0f}, {PW_VEC[i, 1, 0]:0.0f}.\n"

    # Add Target Query instance
    chan_revised_val = (
        (np.log(ch_mat_val[j, 0, :Num_user, :Num_user]) - chan_avg)
        / chan_std
        * 100
    )
    query_text += f"If A is {chan_revised_val[0, 0]:0.0f}, {chan_revised_val[0, 1]:0.0f}, {chan_revised_val[1, 0]:0.0f}, {chan_revised_val[1, 1]:0.0f}, then B is "

    SE_opt, EE_opt = cal_SE_EE(
        ch_mat_val[j, 0, :, :],
        tx_max,
        noise,
        DUE_thr,
        I_thr,
        P_c,
        PW_VEC_val[j],
        opt=critera,
    )

    llm_result = llm(query_text, stop=["\n", "."])["choices"][0]["text"]

    # Parse prediction robustly
    parsed_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", llm_result)
    if len(parsed_numbers) >= 2:
        try:
            temp_PW = np.array(
                [[float(parsed_numbers[0])], [float(parsed_numbers[1])]]
            )
            temp_PW = np.clip(temp_PW, 0, tx_max)
        except ValueError:
            temp_PW = np.zeros((2, 1))
    else:
        temp_PW = np.zeros((2, 1))

    SE_prop, EE_prop = cal_SE_EE(
        ch_mat_val[j, 0, :, :],
        tx_max,
        noise,
        DUE_thr,
        I_thr,
        P_c,
        temp_PW,
        opt=critera,
    )

    # Baselines (Random Continuous and Binary)
    temp_PW_rand = tx_max * np.random.rand(2, 1)
    SE_rand, EE_rand = cal_SE_EE(
        ch_mat_val[j, 0, :, :],
        tx_max,
        noise,
        DUE_thr,
        I_thr,
        P_c,
        temp_PW_rand,
        opt=critera,
    )

    temp_PW_bin = np.zeros((2, 1))
    if np.random.rand() < 0.5:
        temp_PW_bin[0, 0] = tx_max
    else:
        temp_PW_bin[1, 0] = tx_max

    SE_bin, EE_bin = cal_SE_EE(
        ch_mat_val[j, 0, :, :],
        tx_max,
        noise,
        DUE_thr,
        I_thr,
        P_c,
        temp_PW_bin,
        opt=critera,
    )

    if critera == "SE":
        SE_prop_2 = max(SE_bin, SE_prop)
        EE_prop_2 = EE_bin if SE_bin > SE_prop else EE_prop
    else:
        EE_prop_2 = max(EE_bin, EE_prop)
        SE_prop_2 = SE_bin if EE_bin > EE_prop else SE_prop

    SE_opt_mat += SE_opt
    EE_opt_mat += EE_opt * 1000
    SE_prop_mat += SE_prop
    EE_prop_mat += EE_prop * 1000
    SE_prop_2_mat += SE_prop_2
    EE_prop_2_mat += EE_prop_2 * 1000
    SE_rand_mat += SE_rand
    EE_rand_mat += EE_rand * 1000
    SE_bin_mat += SE_bin
    EE_bin_mat += EE_bin * 1000

    if (j + 1) % 10 == 0:
        print(
            f"Index {j+1:03d} | [OPT] SE: {SE_opt_mat/(j+1):0.2f}, EE: {EE_opt_mat/(j+1):0.2f} "
            f"| [PROP] SE: {SE_prop_mat/(j+1):0.2f}, EE: {EE_prop_mat/(j+1):0.2f} "
            f"| [RAND] SE: {SE_rand_mat/(j+1):0.2f}, EE: {EE_rand_mat/(j+1):0.2f}"
        )

print("\n" + "=" * 50)
print("FINAL RESULTS")
print(
    f"[OPT]  SE: {SE_opt_mat/100:0.2f}, EE: {EE_opt_mat/100:0.2f}\n"
    f"[PROP] SE: {SE_prop_mat/100:0.2f}, EE: {EE_prop_mat/100:0.2f}\n"
    f"[RAND] SE: {SE_rand_mat/100:0.2f}, EE: {EE_rand_mat/100:0.2f}"
)
print("=" * 50)