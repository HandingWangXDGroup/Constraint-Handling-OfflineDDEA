# Input:
# dimension             -No. of Decision Variables
# Offlinedata           -offline data
# Eva_num               -Number of evaluations
# lb                    -Lower bound of decision variables
# ub                    -Upper bound of decision variables
# funname               -Test function
#
# Output:
# Execution Time
# optimum           -The final optimal solution.
#
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


import random
import numpy as np
import pandas as pd
import Con_Problem as cp
import time
from RBFN import RBFN
from Latin import latin

import warnings
warnings.filterwarnings("ignore")

class DE():
    def __init__(self, max_iter, dimension, ub, lb, pop_size, F, CR):
        self.pop_size = pop_size  # 种群数量
        self.chrom_length = dimension  # 染色体长度
        self.F = F  # 基础变异算子
        self.CR = CR  # 交叉算子
        self.max_value = ub
        self.min_value = lb
        self.max_iter = max_iter
        self.popfirst = []
        self.fitfirst = []
        self.pop = np.zeros((self.pop_size, self.chrom_length))
        self.init_Population()

    def init_Population(self):
        self.pop = latin(self.pop_size, self.chrom_length, self.min_value, self.max_value)
        # self.pop = np.row_stack((self.pop, x))

    def mutation(self, iter):
        F = self.F
        for seq in range(self.pop_size):
            p1, p2 = random.sample(list(self.pop), 2)
            while (self.pop[seq] == p1).all() or (self.pop[seq] == p2).all():
                p1, p2 = random.sample(list(self.pop), 2)
            newp = self.pop[seq] + F * (p2 - p1) # 生成并取整新个体
            newp = np.max(np.vstack((newp, [self.min_value]*self.chrom_length)), 0)
            newp = np.min(np.vstack((newp, [self.max_value]*self.chrom_length)), 0)
            self.pop = np.row_stack((self.pop, newp))

    def crossover(self):
        for seq in range(self.pop_size):
            newp = np.zeros(self.chrom_length)
            jrand = np.random.randint(0,self.chrom_length)
            for i in range(self.chrom_length):
                if random.random() <= self.CR or i == jrand:
                    newp[i] = self.pop[seq+self.pop_size][i]
                else:newp[i] = self.pop[seq][i]
            newp = np.max(np.vstack((newp, [self.min_value]*self.chrom_length)), 0)
            newp = np.min(np.vstack((newp, [self.max_value]*self.chrom_length)), 0)
            self.pop[seq + self.pop_size] = newp  # 代替变异个体

    def selection(self, fit_value, fit_cons1, fit_cons2):
        newpop = np.zeros((self.pop_size, self.chrom_length))
        for i in range(self.pop_size):
            if fit_cons1[i] >= 0 and fit_cons2[i] >= 0:
                fitnessi1 = fit_value[i]
            else:fitnessi1 = False
            fitnessi2 = np.square(min(0, fit_cons1[i])) + np.square(min(0, fit_cons2[i]))
            if fit_cons1[i + self.pop_size] >= 0 and fit_cons2[i + self.pop_size] >= 0:
                fitnessj1 = fit_value[i + self.pop_size]
            else:fitnessj1 = False
            fitnessj2 = np.square(min(0,fit_cons1[i + self.pop_size])) + np.square(min(0,fit_cons2[i + self.pop_size]))

            if fitnessi1 != False and fitnessj1 != False:
                if fit_value[i] < fit_value[i + self.pop_size]:  # 最小化<, 最大化>
                    newpop[i] = self.pop[i]
                else: newpop[i] = self.pop[i + self.pop_size]
            elif fitnessi1 == False and fitnessj1 != False:
                newpop[i] = self.pop[i + self.pop_size]
            elif fitnessi1 != False and fitnessj1 == False:
                newpop[i] = self.pop[i]
            elif fitnessi2 < fitnessj2:  # 最小化<, 最大化>
                newpop[i] = self.pop[i]
            else: newpop[i] = self.pop[i + self.pop_size]

        cons1_new = TrueCons1.predict(newpop)
        cons2_new = TrueCons2.predict(newpop)
        popa = []
        for i in range(len(newpop)):
            if self.judge(cons1_new[i], cons2_new[i]):
                popa.append(newpop[i])
        if len(popa)>0:
            fitnew = TrueObj.predict(popa)
        else:
            popa = newpop
            fitnew = np.zeros((len(newpop)))
            for j in range(len(newpop)):
                fitnew[i] = np.square(min(0, cons1_new[i])) + np.square(min(0, cons2_new[i]))
        rank = np.argsort(fitnew, axis=0)
        self.fitfirst.append(TrueObj.predict(popa)[rank[0]])  # 最小化0, 最大化-1
        self.popfirst.append(popa[rank[0]])
        firstpop, firstfit = popa[rank[0]], TrueObj.predict(popa)[rank[0]]
        # print('first:', firstpop, 'fit_value:', firstfit)
        self.pop = newpop
        return firstpop, firstfit
    def judge(self, v1, v2):
        if v1>=0 and v2 >= 0:
            return True
        else: return False

class TrueObj():
    def predict(arr):
        '''
        :return: Individual's true objective function value
        '''
        global Eva_num
        if (arr not in dataLibrary):
            Eva_num -= 1
            np.append(dataLibrary,arr)
        return fun(arr)[0]
class TrueCons1():
    def predict(arr):
        '''
        :return: The individual's true first constraint value
        '''
        global Eva_num
        if (arr not in dataLibrary):
            Eva_num -= 1
            np.append(dataLibrary,arr)
        return fun(arr)[1][:,0]
class TrueCons2():
    def predict(arr):
        '''
        :return: The individual's true second constraint value
        '''
        global Eva_num
        if (arr not in dataLibrary):
            Eva_num -= 1
            np.append(dataLibrary,arr)
        return fun(arr)[1][:,1]

if __name__ == '__main__':
    # for funname in ['ellipsoid01','ellipsoid02','rastrigin01','rastrigin02']:
    for funname in ['rastrigin01']:
        for dimension in [10]:
            if funname == 'ellipsoid01':
                fun = cp.ellipsoid01
            elif funname == 'ellipsoid02':
                fun = cp.ellipsoid02
            elif funname == 'rastrigin01':
                fun = cp.rastrigin01
            elif funname == 'rastrigin02':
                fun = cp.rastrigin02
            for testnumber in range(1):
                dataLibrary = latin(N=11 * dimension, D=dimension, lower_bound=-5.12, upper_bound=5.12)
                Eva_num = 10000
                tt0 = time.time()
                max_iter = Eva_num
                de = DE(max_iter=max_iter, dimension=dimension, ub=5.12, lb=-5.12, pop_size=100, F=0.5, CR=0.8)
                de.init_Population()
                EvaList = []
                for iter in range(max_iter):
                    # print('iterator:', iter)
                    de.mutation(iter)
                    de.crossover()
                    fit_value = TrueObj.predict(de.pop)
                    fit_cons1 = TrueCons1.predict(de.pop)
                    fit_cons2 = TrueCons2.predict(de.pop)
                    de.selection(fit_value, fit_cons1, fit_cons2)  # 选择
                    # print('iter:', iter, de.r[iter])
                    # print('True:', fun(de.popfirst[-1]))
                    EvaList.append(Eva_num)
                    if Eva_num <= 0:
                        break
                tt1 = time.time()

                print('fun:', funname, 'dimension:', dimension)
                print('Optimum:', de.popfirst[-1])
                print('True fitness:', fun(de.popfirst[-1]), 'Execute time:', tt1 - tt0)