# Test Functions
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

def ellipsoid01(x):  # x:[-5.12,5.12]
    x = np.array(x)
    if len(x.shape) < 2:
        x = x[np.newaxis, :]
    sum = [0.0] * x.shape[0]
    dimension = x.shape[1]
    cons = np.zeros((x.shape[0], 2))
    for i in range(len(x)):
        for j in range(dimension):
            sum[i] += (j + 1) * np.square(x[i][j])
        cons[i][0] = np.mean(x[i]) - 5/dimension  # >=0
        cons[i][1] = np.sum(x[i][int(dimension/2):])/(dimension-int(dimension/2)) - np.sum(x[i][:int(dimension/2)])/int(dimension/2) # >=0
    sum = np.array(sum)
    return sum, cons

def ellipsoid02(x):  # x:[-5.12,5.12]
    x = np.array(x)
    if len(x.shape) < 2:
        x = x[np.newaxis, :]
    sum = [0.0] * x.shape[0]
    dimension = x.shape[1]
    cons = np.zeros((x.shape[0], 2))
    for i in range(len(x)):
        for j in range(dimension):
            sum[i] += (j + 1) * np.square(x[i][j])
        cons[i][0] = np.sum(np.square(x[i])) - 9*dimension  # >=0
        cons[i][1] = np.sum(np.square(x[i][int(dimension/2):]))/(dimension-int(dimension/2)) - np.sum(np.square(x[i][:int(dimension/2)]))/int(dimension/2) # >=0
    sum = np.array(sum)
    return sum, cons

def rastrigin01(x):  # x:[-5.12,5.12]
    x = np.array(x)
    if len(x.shape) < 2:
        x = x[np.newaxis, :]
    sum = [0.0] * x.shape[0]
    dimension = x.shape[1]
    cons = np.zeros((x.shape[0], 2))
    for i in range(len(x)):
        for j in range(dimension):
            sum[i] += (np.square(x[i][j]) - 10 * np.cos(2 * np.pi * x[i][j]) + 10)
        cons[i][0] = np.mean(x[i]) - 5 / dimension  # >=0
        cons[i][1] = np.sum(x[i][int(dimension/2):])/(int(dimension/2)+1) - np.sum(x[i][:int(dimension/2)])/int(dimension/2) # >=0
    sum = np.array(sum)
    return sum, cons

def rastrigin02(x):  # x:[-5.12,5.12]
    x = np.array(x)
    if len(x.shape) < 2:
        x = x[np.newaxis, :]
    sum = [0.0] * x.shape[0]
    dimension = x.shape[1]
    cons = np.zeros((x.shape[0], 2))
    for i in range(len(x)):
        for j in range(dimension):
            sum[i] += (np.square(x[i][j]) - 10 * np.cos(2 * np.pi * x[i][j]) + 10)
        cons[i][0] = np.sum(np.square(x[i]+1)) - 10 * dimension  # >=0
        cons[i][1] = np.sum(np.square(x[i]-1)) - 10 * dimension  # >=0
    sum = np.array(sum)
    return sum, cons
