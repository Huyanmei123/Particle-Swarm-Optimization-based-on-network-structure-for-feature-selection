# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import os
import sys
import random

from svm import svm_train_2
from mpi4py import MPI
import Graph
import community
import utils

# %%
# data_dir = utils.data_w_dir
# result_dir = utils.result_w_dir
initial_dir = utils.initial_w_dir
Dom_dir = utils.Dom_w_dir
Gbest_dir = utils.Gbest_w_dir
timeEstimation_dir = utils.timeEstimation_w_dir
logpath = os.path.join(utils.log_dir, 'log-PSOFS10_3_w(inertia).txt')

# train_data_path = data_dir+'alizadeh_trainData.csv'
# test_data_path = data_dir + 'alizadeh_testData.csv'
train_data_path =  sys.argv[1]
test_data_path = sys.argv[2]
train_data = pd.read_csv(train_data_path, header=None).values
test_data = pd.read_csv(train_data_path, header=None).values

# %%
def sigmoid(x):
    return 1/(1+math.exp(-x))

def write_rows(path, data, h, m):
    df = pd.DataFrame(data)
    df.to_csv(path, header=h, mode=m)

# %%
class MOPSO:
    def __init__(self, N, dim, Gm, x_min, w, c1, c2, M, part):
        # time1 = time.time()
        self.N = N
        self.dim = dim
        self.Gm = Gm
        self.M = M  # M-objectives problem
        self.part = part
        self.x_min = x_min
        self.c1, self.c2 = c1, c2
        self.w = w
        self.X = np.zeros((2*self.N,self.dim),dtype=float)
        self.V = np.random.random((self.N, self.dim))
        self.Ovs = np.zeros((2*self.N, 6))  # n old pop, n offs
        self.bins = int(self.N * 0.6)
        self.Dm = -np.ones((self.M, self.bins+1), dtype=int)  # sign the nondominate individuals
        self.Fmin = np.zeros(self.M, dtype=float)
        self.Fmax = np.zeros(self.M, dtype=float)
        self.Dom = np.array([])  # 最优解的7个评价指标 
        self.best_pos = np.array([])  # 最优解的个体值
        self.weight = float(sys.argv[5])
        # self.weight = 0.8
        self.timeCost = np.zeros((Gm+1, 3), dtype=float)  # Total time, Search time, svm time
        self.metrics = -np.ones((Gm+1, 6), dtype=float) # tpr, fpr, precission, auc, ratio, acc

    def initialization_netg(self, N, g, c, ver_network, ver_indp, Alpha, Beta, Gamma, ini_p, h):
        time1 = time.time()
        self.X[:N] = np.random.uniform(low=0.0, high=0.5, size=(N, dim)) # 最开始生成的都是小于0.5的，即所有特征都不选
        # 第一次调整：(1) 从网络结构中取5%的节点
        random.shuffle(ver_network)
        f1 = ver_network[: int(0.05*len(ver_network))]
        # (2) 从指定社区中各随机选择一个特征
        # a. 获取所有社区
        coms = c.getCommunities()
        # b. 从社区中取特征
        coms_num = len(coms)
        if coms_num <=10:  # 若不足10个社区，则从每个社区中各随机取一个
            feas = c.getOneNode_perCom()
        else:
            # 随机选中50%的社区，并从每个社区中随机选择1个特征
            index_rand = np.random.choice(np.arange(coms_num), size=int(coms_num*0.5), replace=False)
            index_rand += 1
            fs = c.getOneNode_specified(index_rand)
            feas = np.append(f1, fs)
        self.X[:N, feas] = np.random.uniform(low=0.5, high=1.0, size=(N, len(feas)))
        # 第二次调整：从独立节点中随机选择10%
        indp_sel_num = int(len(ver_indp)*0.1)
        random.shuffle(ver_indp)
        index_rand2 = ver_indp[: indp_sel_num]
        self.X[:N, index_rand2] = np.random.uniform(low=0.5, high=1.0, size=(N, indp_sel_num))
        for i in range(N):
            # 第三次调整：随机选择50%的社区，并随机纠正每个社区中的一个节点
            fea_tocorrct_index = np.random.choice(np.arange(coms_num), size=int(coms_num*0.5), replace=False)+1
            fs2 = c.getOneNode_specified(fea_tocorrct_index)
            for j in fs2:
                u_decode = self.indv_decode(self.X[i])
                # 第j位对应的特征节点的加权度
                djw = g.get_weight_degree(j)
                # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找该特征节点与对应特征节点的weight，并求和
                wjn = g.cal_weight_number(j, u_decode)
                # 该特征节点所在的特征组中被置为1的个数， 在community中找到特征组，并获取该组的特征节点，判断u中的个体元素对应位是否为1
                njc = c.cal_selected_number(j, u_decode)
                pj = Alpha * math.exp(-3/djw) + Beta * math.exp(-wjn) + Gamma * math.exp(-njc)
                if np.random.rand() < sigmoid(pj*random.uniform(-2,2)):
                    self.X[i, j] = np.random.uniform(0.5, 1.0)
        self.up_ini(ini_p, h)
        time2 = time.time()
        totaltime = time2-time1
        self.timeCost[0, :2] = [totaltime, totaltime-self.timeCost[0,-1]]
        return totaltime

    def initialization_network(self, N, g, ver_network, ver_indp, Alpha, Beta, ini_p, h): 
        time1 = time.time()
        self.X[:N] = np.random.uniform(low=0.0, high=0.5, size=(N, dim)) # 最开始生成的都是小于0.5的，即所有特征都不选
        # 第一次调整：从网络中随机选择10%的特征节点
        random.shuffle(ver_network)
        feas = ver_network[: int(0.1*len(ver_network))]
        self.X[:N, feas] = np.random.uniform(low=0.5, high=1.0, size=(N, len(feas)))
        # 第二次调整：从独立节点中随机选择10%
        indp_sel_num = int(len(ver_indp)*0.1)
        random.shuffle(ver_indp)
        index_rand2 = ver_indp[: indp_sel_num]
        self.X[:N, index_rand2] = np.random.uniform(low=0.5, high=1.0, size=(N, indp_sel_num))
        # 第三次调整：通过pj进行纠正（主要是针对网络结构的节点）
        for i in range(N):
            # 从网络中随机选择10%的节点，通过pj进行纠正（特征的选择与否）
            random.shuffle(ver_network)
            sel_vernet = ver_network[: int(0.1*len(ver_network))]
            for j in sel_vernet:
                u_decode = self.indv_decode(self.X[i])
                # 第j位对应的特征节点的加权度
                djw = g.get_weight_degree(j)
                # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找该特征节点与对应特征节点的weight，并求和
                wjn = g.cal_weight_number(j, u_decode)
                pj = Alpha * math.exp(-3/djw) + Beta * math.exp(-wjn)
                if np.random.rand() < sigmoid(pj*random.uniform(-2,2)):
                    self.X[i, j] = np.random.uniform(0.5, 1.0)
        self.up_ini(ini_p, h)
        time2 = time.time()
        totaltime = time2-time1
        self.timeCost[0, :2] = [totaltime, totaltime-self.timeCost[0,-1]]
        return totaltime

    def initialization_random(self, ini_p, h):
        time1 = time.time()
        self.X = np.random.random((2*self.N, self.dim))
        self.up_ini(ini_p, h)
        time2 = time.time()
        totaltime = time2-time1
        self.timeCost[0, :2] = [totaltime, totaltime-self.timeCost[0,-1]]
        return totaltime

    def up_ini(self, ini_p, h):
        [self.obj_fun(i, 0) for i in range(self.N)]
        self.pbest = self.X[:self.N].copy()
        self.pOvs = self.Ovs[:self.N].copy()
        self.updateGbest()
        self.metrics[0] = self.gOvs.copy()
        write_rows(ini_p, self.pOvs, h, 'w')

    def indv_decode(self, x):
        dec_x = x.copy()
        dec_x[dec_x < self.x_min] = 0
        dec_x[dec_x >= self.x_min] = 1
        return dec_x

    def normalization(self, x):
        min_row = np.min(x)  # 找到最小值
        max_row = np.max(x)  # 找到最大值
        _range = max_row - min_row
        return (x-min_row)/_range

    # !目标函数obj_fun
    def obj_fun(self, ith, t):  # _pos_inv 个体的一组元素值，_X DE算法的所有pos值
        f1, f2 = 0.0, 0.0
        metrics = -np.ones(6, dtype=float)
        _pos_inv = self.X[ith]
        dec_indv = self.indv_decode(_pos_inv)
        # obtain the index of dimensionality whose value equal to 1
        feats = np.where(dec_indv == 1)[0]
        counts = len(feats)
        f1 = -counts / self.dim
        if counts!=0:  # calculate f2
            train_features, train_labels = train_data[:, feats], train_data[:, -1]
            test_features, test_labels = test_data[:, feats], test_data[:, -1]
            metrics[:4], f2, svm_t = svm_train_2(train_features, train_labels, test_features, test_labels)
            metrics[-2:] = [f1, f2]
            self.timeCost[t, -1] += svm_t
            self.Ovs[ith] = metrics.copy()
            self.updateDominateset(ith, metrics, _pos_inv)

    # network + community
    def iterator_netg(self, ver_network, g, c, Alpha, Beta, Gamma, vers, percen):
        # 时间开销：total time in one iteration of PSO, time_svm, pj_svm
        for t in range(1, self.Gm+1):
            # 更新velocity和position
            time1 = time.time()
            self.minMax = np.array([np.min(self.X, axis=0), np.max(self.X, axis=0)])  # 独立节点所在维的最大最小值并不会变
            # update the velocity and position of particles
            for i in range(self.N):
                newindex = i+N
                self.r1, self.r2 = np.random.random(2)
                self.V[i] = self.w*self.V[i] + self.c1*self.r1*(self.pbest[i] - self.X[i])+ self.c2*self.r2*(self.gbest - self.X[i])
                # 对网络节点的处理
                random.shuffle(ver_network)
                u_decode = self.indv_decode(self.X[i])
                for j in ver_network:
                    # 第j位对应的特征节点的加权度
                    djw = g.get_weight_degree(j)
                    # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找该特征节点与对应特征节点的weight，并求和
                    wjn = g.cal_weight_number(j, u_decode)
                    if vers[j] != 2: # 若当前节点不在社区中
                    # if j not in ver_com:
                        pj = 0.5 * math.exp(-3/djw) + 0.5 * math.exp(-wjn)
                    else:
                        # 该特征节点所在的特征组中被置为1的个数， 在community中找到特征组，并获取该组的特征节点，判断u中的个体元素对应位是否为1
                        njc = c.cal_selected_number(j, u_decode)
                        # print('djw=', djw, ', wjn=', wjn, ', njc=', njc)
                        pj = Alpha * math.exp(-3/djw) + Beta * math.exp(-wjn) + Gamma * math.exp(-njc)
                    rand_val = np.random.random()
                    if rand_val < pj and self.X[i, j]<self.x_min:
                        if self.V[i, j]>0:
                            self.V[i, j] *= math.exp(pj*percen)
                        else:
                            self.V[i, j] *= math.exp(-pj*percen)
                    elif rand_val >= pj and self.X[i, j]>= self.x_min:
                        if self.V[i, j]>0:
                            self.V[i, j] *= math.exp((pj-1)*percen)
                        else:
                            self.V[i, j] *= math.exp((1-pj)*percen)
                self.X[newindex] += self.V[i]
                self.X[newindex] = self.normalization(self.X[newindex])
                self.obj_fun(newindex, t)
                self.update_pbest(i)
            self.select()
            self.updateGbest()
            self.metrics[t] = self.gOvs.copy()
            time2 = time.time()
            total_time = time2-time1
            self.timeCost[t, :2] = [total_time, total_time-self.timeCost[t, -1]]
            print('Iteration {}, {}'.format(t, self.metrics[t])) 

    # 只有network
    def iterator_network(self, ver_network, g, Alpha, Beta, percen):
        # 时间开销：total time in one iteration of PSO, time_svm, pj_svm
        for t in range(1, self.Gm+1):
            # 更新velocity和position
            time1 = time.time()
            self.minMax = np.array([np.min(self.X, axis=0), np.max(self.X, axis=0)])  # 独立节点所在维的最大最小值并不会变
            # update the velocity and position of particles
            for i in range(self.N):
                newindex = i+N
                self.r1, self.r2 = np.random.random(2)
                self.V[i] = self.w*self.V[i] + self.c1*self.r1*(self.pbest[i] - self.X[i])+ self.c2*self.r2*(self.gbest - self.X[i])
                # 对网络节点做处理
                random.shuffle(ver_network)
                u_decode = self.indv_decode(self.X[i])
                for j in ver_network:
                    # 第j位对应的特征节点的加权度
                    djw = g.get_weight_degree(j)
                    # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找该特征节点与对应特征节点的weight，并求和
                    wjn = g.cal_weight_number(j, u_decode)
                    pj = Alpha * math.exp(-3/djw) + Beta * math.exp(-wjn)
                    rand_val = np.random.random()
                    if rand_val < pj and self.X[i, j]<self.x_min:
                        if self.V[i, j]>0:
                            self.V[i, j] *= math.exp(pj*percen)
                        else:
                            self.V[i, j] *= math.exp(-pj*percen)
                    elif rand_val >= pj and self.X[i, j]>= self.x_min:
                        if self.V[i, j]>0:
                            self.V[i, j] *= math.exp((pj-1)*percen)
                        else:
                            self.V[i, j] *= math.exp((1-pj)*percen)
                self.X[newindex] += self.V[i]
                self.X[newindex] = self.normalization(self.X[newindex])
                self.obj_fun(newindex, t)
                self.update_pbest(i)
            self.select()
            self.updateGbest()
            self.metrics[t] = self.gOvs.copy()
            time2 = time.time()
            total_time = time2-time1
            self.timeCost[t, :2] = [total_time, total_time-self.timeCost[t, -1]]
            print('Iteration {}, {}'.format(t, self.metrics[t])) 

    def iterator_standard(self):
        N = self.N
        for t in range(1, self.Gm+1):
            time1 = time.time()
            # update the velocity and position of particles
            for i in range(N):
                newindex = i+N
                self.r1, self.r2 = np.random.random(2)
                self.V[i] = self.w*self.V[i] + self.c1*self.r1*(self.pbest[i] - self.X[i])+ self.c2*self.r2*(self.gbest - self.X[i])
                self.X[newindex] += self.V[i]
                self.X[newindex] = self.normalization(self.X[newindex])
                self.obj_fun(newindex, t)
                self.update_pbest(i)
            self.select()
            self.updateGbest()
            self.metrics[t] = self.gOvs.copy()
            time2 = time.time()
            total_time = time2-time1
            self.timeCost[t,:2] = [total_time, total_time-self.timeCost[t,-1]]
            print('Iteration {}, {}'.format(t, self.metrics[t]))     

    def select(self):
        N=self.N
        self.reInitialize()
        self.FMaxMin()
        preindex = np.array([], dtype=int)
        selectIndex = np.array([], dtype=int)
        backup = np.array([], dtype=int)      
        # 将值分配到bins个块里面
        for m in range(-self.M, 0):
            unit = (self.Fmax[m] - self.Fmin[m]) / self.bins
            if unit == 0:
                continue
            for i in range(N):
                k = (self.Ovs[i][m] - self.Fmin[m]) / unit
                k = int(k)
                if self.Dm[m][k] == -1:
                    self.Dm[m][k] = i
                elif self.Ovs[i][m] > self.Ovs[self.Dm[m][k]][m]:
                    self.Dm[m][k] = i
                
                k = (self.Ovs[i+N][m] - self.Fmin[m]) / unit
                k = int(k)
                if self.Dm[m][k] == -1:
                    self.Dm[m][k] = i+N
                elif self.Ovs[i+N][m] > self.Ovs[self.Dm[m][k]][m]:
                    self.Dm[m][k] = i+N

        # 取出Dm中!=-1的值
        for m in range(self.M):
            preindex = np.append(preindex, self.Dm[m][self.Dm[m] > -1])
        preindex = list(set(preindex))  # 去重
        if len(preindex) >= N:
            selectIndex = np.random.choice(preindex, N, replace=False)
        else:
            ll = set(np.arange(0, 2*N))
            backup = list(ll.symmetric_difference(set(preindex)))
            surplus = N - len(preindex)
            surplusIndex = np.random.choice(backup, surplus, replace=False)
            selectIndex = np.append(preindex, surplusIndex)
        self.X[:N] = self.X[selectIndex].copy() # update population
        self.Ovs[:N] = self.Ovs[selectIndex].copy()  # update obj vals

    def updateGbest(self):
        size_nondom = len(self.Dom)
        if size_nondom == 1:
            self.gbest = self.best_pos[0].copy()
            self.gOvs = self.Dom[0].copy()
            return 
        # otherwise, divide the nondominated solutions at the first rank into part*part hypercubes
        hypercubes = [[[] for _ in range(self.part)] for i in range(self.part)]
        # nondominated solutions
        nondomX = self.best_pos
        nondomOvs = self.Dom
        minv = [np.min(nondomOvs[:, i-self.M]) for i in range(self.M)]
        maxv = [np.max(nondomOvs[:, i-self.M]) for i in range(self.M)]
        unit = [(maxv[i] - minv[i])/self.part for i in range(self.M)]
        for k in range(len(nondomX)):
            coordinate = [0 for _ in range(self.M)]
            for m in range(self.M):
                coordinate[m-self.M] = math.floor((nondomOvs[k, m-self.M]-minv[m-self.M]) / unit[m-self.M])
                if coordinate[m-self.M] == self.part: # 处理被分在上边界的个体
                    coordinate[m-self.M] = self.part-1
            hypercubes[coordinate[0]][coordinate[1]].append(k) # 正方体记录个体的下标
        probabilities = np.array([[len(hypercubes[i][j]) for j in range(self.part)] for i in range(self.part)])
        probabilities = probabilities.reshape(-1)
        probabilities = probabilities/np.sum(probabilities)
        selCube = np.random.choice(np.arange(len(probabilities)), size=1, replace=False, p=probabilities)[0]
        x, y = selCube//self.part, selCube % self.part
        if(len(hypercubes[x][y])==1):
            self.gbest = nondomX[hypercubes[x][y][0]].copy()
            self.gOvs = nondomOvs[hypercubes[x][y][0]].copy()
        else: # randomly select a solution
            selIndex = np.random.randint(len(hypercubes[x][y]))
            self.gbest = nondomX[hypercubes[x][y][selIndex]].copy()
            self.gOvs = nondomOvs[hypercubes[x][y][selIndex]].copy()
        # return 

    def update_pbest(self, i):
        flag1, flag2 = self.judge_relationship(self.X[i+self.N], self.X[i])
        if flag1 is True and flag2 is True:  # 表示该indv是nondominate solution
            selIndex = np.random.randint(2)
            if selIndex==1:
                self.pbest[i] = self.X[i+self.N].copy()
                self.pOvs[i] = self.Ovs[i+self.N].copy()
        elif flag1 is True and flag2 is False:  # offs is better
            self.pbest[i] = self.X[i+self.N].copy()
            self.pOvs[i] = self.Ovs[i+self.N].copy()
        
    # 更新最优解集
    def updateDominateset(self, ith, metrics, cur_indv):
        flagg = False  # 判断是否可以支配Dom中的解
        flagt = False  # 判断是否为非支配解
        if len(self.Dom) == 0:
            self.Dom = np.array([metrics])
            self.best_pos = np.array([cur_indv])
            return 0 
        delIndex = []
        for d in range(len(self.Dom)):
            flag1, flag2 = self.judge_relationship(self.Ovs[ith], self.Dom[d])
            if flag1 is True and flag2 is True:
                flagt = True  # 表示该indv是nondominate solution
            elif flag1 is True and flag2 is False:
                flagg = True  # 表示该indiv可支配原有解集中的解d，那么应删除原解集中的对应解
                delIndex.append(d)
                flagt = False
            elif flag1 is False:
                # 表明它可以被原解集中的解支配，那么它就不应该加入到解集中
                flagt = False
                flagg = False
                break
        if len(delIndex) > 0:
            self.Dom = np.delete(self.Dom, delIndex, 0)
            self.best_pos = np.delete(self.best_pos, delIndex, 0)
        # 如果该解为非支配解(或者可以支配其他解)，则应该加入到解集中
        if flagg is True or flagt is True:
            self.Dom = np.concatenate((self.Dom, [metrics]), 0)
            self.best_pos = np.concatenate((self.best_pos, [cur_indv]), 0)

    def judge_relationship(self, ovs1, ovs2):
        flag1 = False  # 判断是否有比它大的
        flag2 = False  # 判断是否有比它小的
        for m in range(-self.M, 0):
            if ovs1[m] > ovs2[m]:
                flag1 = True
            elif ovs1[m] < ovs2[m]:
                flag2 = True
        return flag1, flag2

    # 找到self.Ovs中的每个目标的最大最小函数值
    def FMaxMin(self):
        for m in range(-self.M, 0):
            # find the max and min in every objective
            self.Fmin[m] = np.min(self.Ovs[:, m])
            self.Fmax[m] = np.max(self.Ovs[:, m])
    
    # reset
    def reInitialize(self):
        self.Dm = -np.ones((self.M, self.bins+1), dtype=int)
  

    def execution(self, dataset):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # rank=0
        g, c = None, None
        Alpha, Beta, Gamma = float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])
        path_network = sys.argv[3]
        path_com = sys.argv[4]
        flag = sys.argv[11] # 3代表NetG-PSO, 2代表Net-PSO, 1代表PSO
        percen = float(sys.argv[13])
        # Alpha, Beta, Gamma = 0.3, 0.3, 0.4
        # path_network = './data/alizadeh-2000-v1-norm-pearson06.txt'
        # path_com = './data/alizadeh-norm-com0.80.txt'
        # flag = '3'
        # percen = 0.2
        timestamp = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
        fit_h = ['Tpr', 'Fpr', 'Precission', 'Auc', 'Ratio', 'Accuracy']
        time_h = ['Total_t', 'Search_t', 'Svm_t']

        if flag=='1':
            algorithm = 'MOPSO1'
            print('rank %d running %s for %s' % (rank, algorithm, dataset))
            str1 = '{} {} weight_{} Iteration_{} N_{} Percen_{} w_{} rank_{} {}'.format(dataset, algorithm, self.weight, self.Gm, self.N, percen, self.w, rank, timestamp)
            init_time = self.initialization_random(initial_dir+'Initialization '+str1+'.csv', fit_h)
            dom_path = '{}Dom {}.csv'.format(Dom_dir, str1)
            self.iterator_standard()
        elif flag=='2':
            algorithm='Net-MOPSO'
            g = Graph.Graph(path_network, self.weight)
            ver_network = list(g.getVertices())
            vers = np.zeros(self.dim, dtype=int)
            vers[ver_network] = 1
            ver_indp = np.where(vers==0)[0]
            print('rank %d running %s for %s, weight=%.2f' % (rank, algorithm, dataset, self.weight))
            # Alpha, Beta = 0.5, 0.5
            str1 = '{} {} weight_{} Iteration_{} N_{} Percen_{} w_{} rank_{} {}'.format(dataset, algorithm, self.weight, self.Gm, self.N, percen, self.w, rank, timestamp)
            init_time = self.initialization_network(self.N, g, ver_network, ver_indp, Alpha, Beta, initial_dir+'Initialization '+str1+'.csv', fit_h)
            # Ovs_p = ovs_dir + 'Ovs ' + str1 + '.csv'
            self.iterator_network(ver_network, g, Alpha, Beta, percen)
        else:
            algorithm = 'NetG-MOPSO'
            g = Graph.Graph(path_network, self.weight)
            c = community.CommunityGroup(path_com)
            ver_network = list(g.getVertices())  # 网络结构中的节点
            ver_com = list(c.getAllVertices())  # 社区中的节点
            vers = np.zeros(self.dim, dtype=int)
            vers[ver_network] = 1
            vers[ver_com] = 2
            ver_indp = np.where(vers==0)[0]  # 独立节点（不在网络结构中的节点）
            print('rank %d running %s for %s, weight=%.2f' % (rank, algorithm, dataset, self.weight))
            # Alpha, Beta, Gamma = 0.3, 0.3, 0.4
            str1 = '{} {} weight_{} Iteration_{} N_{} Percen_{} w_{} rank_{} {}'.format(dataset, algorithm, self.weight, self.Gm, self.N, percen, self.w, rank, timestamp)
            init_time = self.initialization_netg(self.N, g, c, ver_network, ver_indp, Alpha, Beta, Gamma, initial_dir+'Initialization '+str1+'.csv', fit_h)
            self.iterator_netg(ver_network, g, c, Alpha, Beta, Gamma, vers, percen)
        # write results to csv files
        dom_path = '{}Dom {}.csv'.format(Dom_dir, str1)
        gbest_path = '{}Gbest {}.csv'.format(Gbest_dir, str1)
        time_path = '{}timeEstimation {}.csv'.format(timeEstimation_dir, str1)
        
        f = open(logpath+'log-MOPSO(10-3).txt', 'a')
        f.write(algorithm+' initialization cost '+str(init_time)+' seconds\n')
        f.close()
        write_rows(dom_path, self.Dom, fit_h, 'w')
        write_rows(gbest_path, self.metrics, fit_h, 'w')
        write_rows(time_path, self.timeCost, time_h, 'w')
        print('rank {} finished {} for {}'.format(rank, algorithm, dataset))


#%%
if __name__ == "__main__":
    np.random.seed(int(np.random.rand()*100000))
    dim = len(train_data[0])-1
    dataset = sys.argv[10]
    N = int(sys.argv[12])
    Gm=int(sys.argv[9])
    # dataset = 'Alizadeh'
    # N = 200
    # Gm = 50
    x_min = 0.5
    lp, up = 0, 1
    c1, c2 = 2, 2
    w = float(sys.argv[14])
    M = 2 # 2 objs, accuracy, ratio
    part=3
    mopso = MOPSO(N, dim, Gm, x_min, w, c1, c2, M, part)
    mopso.execution(dataset)

