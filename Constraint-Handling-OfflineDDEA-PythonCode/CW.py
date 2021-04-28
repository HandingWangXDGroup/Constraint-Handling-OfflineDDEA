# Input:
# dimension             -No. of Decision Variables
# Offlinedata           -offline data
# Eva_num               -Number of evaluations
# lower_bound           -Lower bound of decision variables
# upper_bound           -Upper bound of decision variables
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

import time
import random
import numpy as np
import pandas as pd
import Con_Problem as cp
import time
from Latin import latin

import warnings
warnings.filterwarnings("ignore")

class MOE():
    def __init__(self,pop_size,dimension, ub, lb):
        self.pop_size = pop_size
        self.max_value = ub
        self.min_value = lb
        self.chrom_length = dimension
        self.mu = self.chrom_length + 1
        self.popfirst = []
        self.A = []
        self.pop = np.zeros((self.pop_size, self.chrom_length))

    def init_Population(self):
        self.pop = latin(self.pop_size, self.chrom_length, self.min_value, self.max_value)

    def ndsort(self, poparr, fit_value, fit_cons1, fit_cons2):
        '''
        :param poparr: Population
        :param fit_value: The fitness of the individual on the objective function
        :param fit_cons1: The fitness of the individual on the first constraint
        :param fit_cons2: The fitness of the individual on the second constraint
        :return: Non-dominated individuals
        '''
        temp1 = np.min(np.vstack((fit_cons1, np.array([0] * len(fit_value)))), 0)
        temp2 = np.min(np.vstack((fit_cons2, np.array([0] * len(fit_value)))), 0)
        temp = np.square(temp1)+np.square(temp2)
        fit_all = np.column_stack((fit_value,temp))
        current_front = []
        domiarr = np.zeros((len(fit_all)))
        for i in range(len(fit_all)-1):
            for j in range(i+1,len(fit_all)):
                if dominate(fit_all[i],fit_all[j]):
                    domiarr[j] += 1
                elif dominate(fit_all[j],fit_all[i]):
                    domiarr[i] += 1
        for i in range(len(fit_all)):
            if domiarr[i] == 0:
                current_front.append(poparr[i])
        temp = np.array(current_front)
        return temp

    def condition1(self):
        theta1 = 1e-10
        fit_value = TrueObj.predict(self.pop)
        fit_cons1 = TrueCons1.predict(self.pop)
        fit_cons2 = TrueCons2.predict(self.pop)
        arr = []
        for i in range(len(fit_value)):
            if fit_cons1[i] >= 0 and fit_cons2[i] >= 0:
                arr.append(fit_value[i])
        if len(arr) > 0:
            if max(arr)-min(arr)<theta1:
                return True
            else:
                return False
        else:
            return False

    def condition2(self):
        theta2 = 12
        fit_value = TrueObj.predict(self.pop)
        fit_cons1 = TrueCons1.predict(self.pop)
        fit_cons2 = TrueCons2.predict(self.pop)
        for i in range(len(fit_value)):
            if fit_cons1[i] >= 0 and fit_cons2[i] >= 0:
                return False
        if max(fit_value)-min(fit_value) < theta2:
            return True
        return False

    def SPXcrossover(self,Q):
        spxnum = 10
        simplex_factor = int(dimension/5 + 5)
        newpop = []
        qarr = np.mean(Q,axis=0)
        arr = Q-qarr
        for i in range(spxnum):
            k = np.random.uniform(0, 1, self.mu)
            k_ = sum(k)
            k = k / k_
            y = np.matmul(simplex_factor * k, arr) + qarr
            y = np.max(np.vstack((y, np.array([self.min_value] * self.chrom_length))), 0)
            y = np.min(np.vstack((y, np.array([self.max_value] * self.chrom_length))), 0)
            newpop.append(y)
        return np.array(newpop)

    def model_1(self):
        Qindex = random.sample([int(x) for x in range(len(self.pop))], self.mu)
        Q = self.pop[Qindex]
        self.pop = np.delete(self.pop, Qindex, axis=0)
        C = self.SPXcrossover(Q)
        cfit1 = TrueObj.predict(C)
        cfit2 = TrueCons1.predict(C)
        cfit3 = TrueCons2.predict(C)
        R = self.ndsort(C, cfit1, cfit2, cfit3)
        qfit = TrueObj.predict(Q)
        for i in range(len(R)):
            arr = find(R[i], Q)
            if sum(arr) == 0:
                continue
            elif sum(arr) == 1:
                weizhi = np.where(arr == 1)
                Q[weizhi] = R[i]
                qfit[weizhi] = TrueObj.predict([R[i]])
            elif TrueCons1.predict([R[i]])>=0 and TrueCons2.predict([R[i]])>=0:
                index = np.where(arr == 1)
                afit = qfit[index]
                weizhi = np.where(afit == max(afit))
                Q[index[0][weizhi]] = R[i]
                qfit[index[0][weizhi]] = TrueObj.predict([R[i]])
            else:
                index = np.where(arr == 1)
                weizhi = np.random.randint(0,len(index[0]))
                Q[index[0][weizhi]] = R[i]
                qfit[index[0][weizhi]] = TrueObj.predict([R[i]])
        self.pop = np.row_stack((self.pop, Q))

    def model_2(self):
        Qindex = random.sample([int(x) for x in range(len(self.pop))], self.mu)
        Q = self.pop[Qindex]
        self.pop = np.delete(self.pop, Qindex, axis=0)
        C = self.SPXcrossover(Q)
        cfit1 = TrueObj.predict(C)
        cfit2 = TrueCons1.predict(C)
        cfit3 = TrueCons2.predict(C)
        R = self.ndsort(C, cfit1, cfit2, cfit3)
        qfit = TrueObj.predict(Q)
        x_1 = random.choice(R)
        arr = find(x_1, Q)
        for i in range(1):
            if sum(arr) == 0:
                continue
            elif sum(arr) == 1:
                weizhi = np.where(arr == 1)
                Q[weizhi] = R[i]
            elif TrueCons1.predict([x_1]) >= 0 and TrueCons2.predict([x_1]) >= 0:
                index = np.where(arr==1)
                afit = qfit[index]
                weizhi = np.where(afit == max(afit))
                Q[index[0][weizhi]] = x_1
                qfit[index[0][weizhi]] = TrueObj.predict([x_1])
            else:
                index = np.where(arr==1)
                weizhi = np.random.randint(0,len(index[0]))
                Q[index[0][weizhi]] = x_1
                qfit[index[0][weizhi]] = TrueObj.predict([x_1])
        self.pop = np.row_stack((self.pop, Q))

    def model_3(self):
        Qindex = random.sample([int(x) for x in range(len(self.pop))], self.mu)
        Q = self.pop[Qindex]
        self.pop = np.delete(self.pop, Qindex, axis=0)
        C = self.SPXcrossover(Q)
        cfit1 = TrueObj.predict(C)
        cfit2 = TrueCons1.predict(C)
        cfit3 = TrueCons2.predict(C)
        R = self.ndsort(C, cfit1, cfit2, cfit3)
        rfit1 = TrueObj.predict(R)
        rfit2 = TrueCons1.predict(R)
        rfit3 = TrueCons2.predict(R)
        qfit1 = TrueObj.predict(Q)
        qfit2 = TrueCons1.predict(Q)
        qfit3 = TrueCons2.predict(Q)
        for i in range(len(R)):
            for j in range(len(Q)):
                if judge(rfit2[i], rfit3[i]) == True and judge(qfit2[j], qfit3[j]) == True:
                    if rfit1[i] < qfit1[j]:
                        Q[j] = R[i]
                elif judge(rfit2[i], rfit3[i]) == True and judge(qfit2[j], qfit3[j]) == False:
                    Q[j] = R[i]
                elif judge(rfit2[i], rfit3[i]) == False and judge(qfit2[j], qfit3[j]) == False:
                    if np.square(min(0, rfit2[i])) + np.square(min(0, rfit3[i])) < np.square(min(0, qfit2[j])) + np.square(min(0, qfit3[j])):
                        Q[j] = R[i]
        self.pop = np.row_stack((self.pop, Q))

    def ISRAM(self, iter):
        m = 10
        pfit1 = TrueObj.predict(self.pop)
        pfit2 = TrueCons1.predict(self.pop)
        pfit3 = TrueCons2.predict(self.pop)
        tempp = []
        tempc = []
        for i in range(len(pfit1)):
            if pfit2[i] < 0 or pfit3[i] < 0:
                tempp.append(self.pop[i])
                tempc.append([np.square(min(0, pfit2[i])) + np.square(min(0, pfit3[i]))])
        if len(tempp)>0:
            tempp = np.array(tempp)
            x_i = tempp[np.where(tempc == min(tempc))]

        if self.condition1() == False:
            if len(tempp) == len(self.pop):
                self.A = np.append(self.A, x_i)
            if iter%m == 0:
                self.A = np.unique(self.A)
                huan_num = min(len(self.A), 2)
                if huan_num > 0:
                    new_individuals = random.sample(self.A, huan_num)
                    seq = random.sample([int(x) for x in range(len(self.pop))], self.mu)
                    for i in range(len(seq)):
                        self.pop[seq[i]] = new_individuals[i]
                    self.A = []

    def savefirst(self):
        '''
        Save the best individuals in the population, and choose feasible solutions first, and then the solution with the best fitness.
        '''
        fit_value = TrueObj.predict(self.pop)
        fit_cons1 = TrueCons1.predict(self.pop)
        fit_cons2 = TrueCons2.predict(self.pop)
        feasInd = []
        feasFit = []
        InfeasInd = []
        InfeasFit = []
        for j in range(len(self.pop)):
            if fit_cons1[j] >= 0 and fit_cons2[j] >= 0:
                feasInd.append(self.pop[j])
                feasFit.append(fit_value[j])
            else:
                InfeasInd.append(self.pop[j])
                InfeasFit.append(np.square(min(0, fit_cons1[j])) + np.square(min(0, fit_cons2[j])))
        if len(feasInd) > 0:
            feasInd = np.array(feasInd)
            feasFit = np.array(feasFit)
            first_individual = feasInd[np.where(feasFit == min(feasFit))]
            self.popfirst.append(first_individual.reshape((self.chrom_length)))
        else:
            InfeasInd = np.array(InfeasInd)
            InfeasFit = np.array(InfeasFit)
            first_individual = InfeasInd[np.where(InfeasFit == min(InfeasFit))]
            self.popfirst.append(first_individual.reshape((self.chrom_length)))

def judge(v1, v2):
    '''
    Determine whether both constraints are satisfied
    '''
    if v1 >= 0 and v2 >= 0:
        return True
    else: return False

def find(arr, data):
    arr = arr[np.newaxis,:]
    temp = np.zeros((len(data)))
    t1 = np.append(TrueObj.predict(arr), np.square(min(0, TrueCons1.predict(arr))) + np.square(min(0, TrueCons2.predict(arr))))
    for i in range(len(data)):
        t2 = np.column_stack((TrueObj.predict(data[i][np.newaxis, :]), np.square(min(0, TrueCons1.predict(data[i][np.newaxis, :])) + np.square(min(0, TrueCons2.predict(data[i][np.newaxis, :])))))).ravel()
        if dominate(t1, t2):
            temp[i] = 1
    return temp

def dominate(ind1, ind2):
    '''
    Determine whether ind1 dominates ind2
    :param ind1:
    :param ind2:
    :return: True or False
    '''
    for a, b in zip(ind1, ind2):
        if a > b:
            return False
    return True

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
    '''
    :return: The individual's true second constraint value
    '''
    def predict(arr):
        global Eva_num
        if (arr not in dataLibrary):
            Eva_num -= 1
            np.append(dataLibrary,arr)
        return fun(arr)[1][:,1]

if __name__ == '__main__':
    # for funname in ['ellipsoid01','ellipsoid02','rastrigin01','rastrigin02']:
    for funname in ['ellipsoid01']:
        for dimension in [10]:
            if funname == 'ellipsoid01':
                fun = cp.ellipsoid01
            elif funname == 'ellipsoid02':
                fun = cp.ellipsoid02
            elif funname == 'rastrigin01':
                fun = cp.rastrigin01
            elif funname == 'rastrigin02':
                fun = cp.rastrigin02
            for xxx in range(1):

                dataLibrary = latin(N=11*dimension, D=dimension, lower_bound=-5.12, upper_bound=5.12)
                Eva_num = 10000
                tt0 = time.time()
                max_iter = Eva_num
                moe = MOE(pop_size=100, dimension=dimension, ub=5.12, lb=-5.12)
                moe.init_Population()
                EvaList = []
                for iter in range(max_iter):
                    if moe.condition2() == False:
                        if random.choice([1, 2]) == 1:
                            moe.model_1()
                        else:
                            moe.model_2()
                    else:
                        # print('condition2 True!')
                        moe.model_3()
                    if moe.condition2() == False:
                        moe.ISRAM(iter)
                    moe.savefirst()
                    EvaList.append(Eva_num)
                    # print('iter:', iter)
                    # print('True:', fun(moe.popfirst[-1]))
                    if Eva_num <= 0:
                        break
                tt1 = time.time()
                moe.popfirst = np.array(moe.popfirst)
                print('fun:', funname, 'dimension:', dimension)
                print('Optimum:', moe.popfirst[-1])
                print('True fitness:', fun(moe.popfirst[-1]), 'Execute time:', tt1 - tt0)
