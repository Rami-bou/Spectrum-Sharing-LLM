import math
import random
import numpy as np
import itertools

data = []
def data_gen(num_samples):
    count = 0
    while (count < num_samples):
        tx1 = [random.uniform(0, 30), random.uniform(0, 50)]
        d = random.uniform(5, 10)
        theta = random.uniform(0, 2 * math.pi)
        rx1 = [tx1[0] + d * math.cos(theta), tx1[1] + d * math.sin(theta)]

        tx2 = [random.uniform(5, 10), random.uniform(0, 50)]
        d = random.uniform(5, 10)
        theta = random.uniform(0, 2 * math.pi)
        rx2 = [tx2[0] + d * math.cos(theta), tx2[1] + d * math.sin(theta)]

        tx3 = [random.uniform(8, 20), random.uniform(0, 50)]
        d = random.uniform(5, 10)
        theta = random.uniform(0, 2 * math.pi)
        rx3 = [tx3[0] + d * math.cos(theta), tx3[1] + d * math.sin(theta)]

        tx4 = [random.uniform(10, 30), random.uniform(0, 50)]
        d = random.uniform(5, 10)
        theta = random.uniform(0, 2 * math.pi)
        rx4 = [tx4[0] + d * math.cos(theta), tx4[1] + d * math.sin(theta)]

        tx5 = [random.uniform(15, 30), random.uniform(0, 50)]
        d = random.uniform(5, 10)
        theta = random.uniform(0, 2 * math.pi)
        rx5 = [tx5[0] + d * math.cos(theta), tx5[1] + d * math.sin(theta)]

        d11 = np.sqrt((tx1[0]-rx1[0])**2 + (tx1[1]-rx1[1])**2)
        d12 = np.sqrt((tx1[0]-rx2[0])**2 + (tx1[1]-rx2[1])**2)
        d13 = np.sqrt((tx1[0]-rx3[0])**2 + (tx1[1]-rx3[1])**2)
        d14 = np.sqrt((tx1[0]-rx4[0])**2 + (tx1[1]-rx4[1])**2)
        d15 = np.sqrt((tx1[0]-rx5[0])**2 + (tx1[1]-rx5[1])**2)

        d21 = np.sqrt((tx2[0]-rx1[0])**2 + (tx2[1]-rx1[1])**2)
        d22 = np.sqrt((tx2[0]-rx2[0])**2 + (tx2[1]-rx2[1])**2)
        d23 = np.sqrt((tx2[0]-rx3[0])**2 + (tx2[1]-rx3[1])**2)
        d24 = np.sqrt((tx2[0]-rx4[0])**2 + (tx2[1]-rx4[1])**2)
        d25 = np.sqrt((tx2[0]-rx5[0])**2 + (tx2[1]-rx5[1])**2)

        d31 = np.sqrt((tx3[0]-rx1[0])**2 + (tx3[1]-rx1[1])**2)
        d32 = np.sqrt((tx3[0]-rx2[0])**2 + (tx3[1]-rx2[1])**2)
        d33 = np.sqrt((tx3[0]-rx3[0])**2 + (tx3[1]-rx3[1])**2)
        d34 = np.sqrt((tx3[0]-rx4[0])**2 + (tx3[1]-rx4[1])**2)
        d35 = np.sqrt((tx3[0]-rx5[0])**2 + (tx3[1]-rx5[1])**2) 

        d41 = np.sqrt((tx4[0]-rx1[0])**2 + (tx4[1]-rx1[1])**2)
        d42 = np.sqrt((tx4[0]-rx2[0])**2 + (tx4[1]-rx2[1])**2)
        d43 = np.sqrt((tx4[0]-rx3[0])**2 + (tx4[1]-rx3[1])**2)
        d44 = np.sqrt((tx4[0]-rx4[0])**2 + (tx4[1]-rx4[1])**2)
        d45 = np.sqrt((tx4[0]-rx5[0])**2 + (tx4[1]-rx5[1])**2)

        d51 = np.sqrt((tx5[0]-rx1[0])**2 + (tx5[1]-rx1[1])**2)
        d52 = np.sqrt((tx5[0]-rx2[0])**2 + (tx5[1]-rx2[1])**2)
        d53 = np.sqrt((tx5[0]-rx3[0])**2 + (tx5[1]-rx3[1])**2)
        d54 = np.sqrt((tx5[0]-rx4[0])**2 + (tx5[1]-rx4[1])**2)
        d55 = np.sqrt((tx5[0]-rx5[0])**2 + (tx5[1]-rx5[1])**2)

        h11 = 1 / d11**3
        h12 = 1 / d12**3
        h13 = 1 / d13**3
        h14 = 1 / d14**3
        h15 = 1 / d15**3

        h21 = 1 / d21**3
        h22 = 1 / d22**3
        h23 = 1 / d23**3
        h24 = 1 / d24**3
        h25 = 1 / d25**3

        h31 = 1 / d31**3
        h32 = 1 / d32**3
        h33 = 1 / d33**3
        h34 = 1 / d34**3
        h35 = 1 / d35**3

        h41 = 1 / d41**3
        h42 = 1 / d42**3
        h43 = 1 / d43**3
        h44 = 1 / d44**3
        h45 = 1 / d45**3

        h51 = 1 / d51**3
        h52 = 1 / d52**3
        h53 = 1 / d53**3
        h54 = 1 / d54**3
        h55 = 1 / d55**3
        count += 1
        data.append([[h11, h12, h13, h14, h15], 
                    [h21, h22, h23, h24, h25], 
                    [h31, h32, h33, h34, h35], 
                    [h41, h42, h43, h44, h45], 
                    [h51, h52, h53, h54, h55]])

        return data


P = []
I = []
SE = []
SINR = []

def calc_best_power(data):

    for H in data:

        best_se = -np.inf
        best_power = None
        best_sinr = None
        best_interference = None
        best_se_list = None

        # Every combination of P1,P2,P3,P4,P5
        for powers in itertools.product(range(1, 21), repeat=5):

            interferences = []
            sinrs = []
            ses = []

            for t in range(5):

                signal = H[t][t] * powers[t]

                inter = 0.0
                for k in range(5):
                    if k != t:
                        inter += powers[k] * H[k][t]

                sinr = signal / (1.0 + inter)
                se = np.log2(1 + sinr)

                interferences.append(inter)
                sinrs.append(sinr)
                ses.append(se)

            total_se = sum(ses)

            if total_se > best_se:
                best_se = total_se
                best_power = list(powers)
                best_sinr = sinrs
                best_interference = interferences
                best_se_list = ses

        P.append(best_power)
        I.append(best_interference)
        SINR.append(best_sinr)
        SE.append(best_se_list)

    return P, I, SINR, SE

def build_icl(length):
    #for _ in range(length):
    data = data_gen(1)
    P, I, SINR, SE = calc_best_power(data)
    print(data)
    prompt = """Allocation Examples:\n"""
    for i in range(5):
        for j in range(5):
            prompt += f"""If {data[0][i][j]}, """
        prompt += """ then {P[0][i]}\n"""

    return prompt

print(build_icl(1))