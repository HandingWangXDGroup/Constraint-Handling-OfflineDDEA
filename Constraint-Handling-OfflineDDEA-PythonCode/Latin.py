# Latin hypercube sampling
# N - The size of the sample data
# D - No.of Decision Variables
####    Authors:    Pengfei Huang, Handing Wang
####    Xidian University, China
####    EMAIL:      pfeihuang@foxmail.com, hdwang@xidian.edu.cn
####    DATE:       April 2021
# ------------------------------------------------------------------------
# This code is part of the program that produces the results in the following paper:
#
# Pengfei Huang,Handing Wang,Comparative Empirical Study on Constraint Handling in Offline Data-Driven Evolutionary Optimization.
#
# You are free to use it for non-commercial purposes. However, we do not offer any forms of guanrantee or warranty associated with the code. We would appreciate your acknowledgement.
# ------------------------------------------------------------------------

import numpy as np

def latin(N ,D,lb,ub):
    d = 1.0/N
    result = np.empty([N, D])
    temp = np.empty([N])
    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size = 1)[0]

        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    lower_bounds = lb
    upper_bounds = ub
    if np.any(lower_bounds > upper_bounds):
        return None
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,
                       (upper_bounds - lower_bounds),
                       out=result),
           lower_bounds,
           out=result)
    return result
