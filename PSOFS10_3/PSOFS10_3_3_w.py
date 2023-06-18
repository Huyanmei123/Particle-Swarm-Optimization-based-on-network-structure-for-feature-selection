#%%
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
import Graph1
import community
import utils

#%%
initial_dir = utils.initial_w_dir
excels_dir = utils.excels_w_dir
timeEstimation_dir = utils.timeEstimation_w_dir
logpath = os.path.join(utils.log_dir, 'log-PSOFS10_3_w(inertia).txt')


#%%
# train, test data
def read_raw_data(path, h):
    data = pd.read_csv(path, header=h)
    return data.values

def write_rows(data, path, h, m):
    df = pd.DataFrame(data)
    df.to_csv(path, header=h, mode=m)

# train_path = '../../data/shipp_trainData.csv'
# test_path = '../../data/shipp_testData.csv'
train_path = sys.argv[1]
test_path = sys.argv[2]
trainData = read_raw_data(train_path, None)
testData = read_raw_data(test_path, None)

#%%
def sigmoid(x):
    return 1/(1+math.exp(-x))
    
#%%
# 画图
def draw_lineChart(y, ylabel, dataset, algorithm, path):
    plt.clf()
    x = np.arange(len(y))
    plt.plot(x, y, label=algorithm)
    plt.title(dataset)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(path)

#%%
class PSO:
    def __init__(self, pN, dim, Gm, w, c1, c2):
        self.N, self.dim = pN, dim
        self.Gm = Gm
        self.w = w
        self.x_min = 0.5
        # self.weight = 0.75
        self.weight = float(sys.argv[5])
        self.c1, self.c2 = c1, c2
        self.timeCost = np.zeros((Gm+1, 3), dtype=float)  # total time, search time, svm time
        self.V = np.random.random((pN, dim))  # 粒子的初始速度是[0, 1]随机小数
        self.Ovs = -np.ones((pN, 7), dtype=float)
        # self.p_ovs = np.array([self.obj_fun(self.X[i], 0) for i in range(self.N)])  # 记录每次迭代中每个个体的两个目标值，以及适应度值, (n, 7)的矩阵
        self.gBest = np.zeros(dim, dtype=float)
        self.g_ovs = -np.ones(7, dtype=float)
        self.metrics = -np.ones((Gm+1, 7), dtype=float)  # 记录每次迭代中最好个体的metrics取值 tpr, fpr, precission, auc, f1, f2, fitness
  
    def initialization_netg(self, N, g, c, ver_network, ver_indp, Alpha, Beta, Gamma, ini_p, h):
        time1 = time.time()
        self.X = np.random.uniform(low=0.0, high=0.5, size=(N, dim)) # 最开始生成的都是小于0.5的，即所有特征都不选
        # 第一次调整：(1) 从网络结构中取5%的节点
        random.shuffle(ver_network)
        f1 = ver_network[: int(0.05*len(ver_network))]
        # (2) 从指定社区中各随机选择一个特征
        # a. 获取所有社区
        coms = c.getCommunities()
        # b. 从社区中取特征
        coms_num = len(coms)
        if coms_num <=10: # 若不足10个社区，则从每个社区中各随机取一个
            feas = c.getOneNode_perCom()
        else:
            # 随机选中50%的社区，并从每个社区中随机选择1个特征
            index_rand = np.random.choice(np.arange(coms_num), size=int(coms_num*0.5), replace=False)
            index_rand += 1
            fs = c.getOneNode_specified(index_rand)
            feas = np.append(f1, fs)
        self.X[:, feas] = np.random.uniform(low=0.5, high=1.0, size=(N, len(feas)))
        # 第二次调整：从独立节点中随机选择10%
        indp_sel_num = int(len(ver_indp)*0.1)
        random.shuffle(ver_indp)
        index_rand2 = ver_indp[: indp_sel_num]
        self.X[:, index_rand2] = np.random.uniform(low=0.5, high=1.0, size=(N, indp_sel_num))
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

    def initialization_network(self, N, g, ver_network, ver_indp, Alpha, Beta, ini_p, h): 
        time1 = time.time()
        self.X = np.random.uniform(low=0.0, high=0.5, size=(N, dim)) # 最开始生成的都是小于0.5的，即所有特征都不选
        # 第一次调整：从网络中随机选择10%的特征节点
        random.shuffle(ver_network)
        feas = ver_network[: int(0.1*len(ver_network))]
        self.X[:, feas] = np.random.uniform(low=0.5, high=1.0, size=(N, len(feas)))
        # 第二次调整：从独立节点中随机选择10%
        indp_sel_num = int(len(ver_indp)*0.1)
        random.shuffle(ver_indp)
        index_rand2 = ver_indp[: indp_sel_num]
        self.X[:, index_rand2] = np.random.uniform(low=0.5, high=1.0, size=(N, indp_sel_num))
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

    def initialization_random(self, ini_p, h):
        print('initialization!')
        time1 = time.time()
        self.timeCost = np.zeros((self.Gm+1, 3), dtype=float)  # total time, search time, svm time
        self.X = np.random.random((self.N, self.dim))
        self.up_ini(ini_p, h)
        time2 = time.time()
        totaltime = time2-time1
        self.timeCost[0, :2] = [totaltime, totaltime-self.timeCost[0,-1]]

    def up_ini(self, ini_p, h):
        self.Ovs = np.array([self.obj_fun(self.X[i], 0) for i in range(self.N)])
        self.pBest = self.X.copy()
        self.p_ovs = self.Ovs.copy()
        maxIndex = np.argmax(self.Ovs[:, -1])  # 找最大目标函数值
        self.gBest = self.pBest[maxIndex].copy()
        self.g_ovs = self.p_ovs[maxIndex].copy()
        self.metrics[0] = self.g_ovs.copy()
        write_rows(self.p_ovs, ini_p, h, 'w')

    def indv_decode(self, x):
        x_decode = x.copy()
        x_decode[x>=self.x_min]=1
        x_decode[x<self.x_min]=0
        return x_decode
    
    def normalization(self, x):
        min_row = np.min(x)  # 找到最小值
        max_row = np.max(x)  # 找到最大值
        _range = max_row - min_row
        return (x-min_row)/_range

    def obj_fun(self, x, t):
        f1, f2, fitness_value = 0.0, 0.0, 0.0
        Alpha, Beta = -0.3, 0.7
        metrics = -np.ones(7, dtype=float)
        feats = []
        decode_indv = self.indv_decode(x)
        feats = np.where(decode_indv==1)[0]
        count = np.sum(decode_indv==1)
        f1 = count/self.dim
        if len(feats)>0:
            train_features = trainData[:, feats]
            train_labels = trainData[:, -1]
            test_features = testData[:, feats]
            test_labels = testData[:, -1]
            metrics[:4], f2, svm_time = svm_train_2(train_features, train_labels, test_features, test_labels)
            # f2 = 1-f2  # error rate
            self.timeCost[t, -1] += svm_time
        # print('f1=',f1, 'f2=', f2)
        fitness_value = Alpha*f1 + Beta*f2
        metrics[4:] = [f1, f2, fitness_value]
        return metrics
    
    def iterator_standard(self):
        for t in range(1, self.Gm+1):
            time1 = time.time()
            # 更新velocity和position
            for i in range(self.N):
                self.r1, self.r2 = np.random.random(2)
                self.V[i] = self.w*self.V[i] + self.c1*self.r1*(self.pBest[i] - self.X[i])+ self.c2*self.r2*(self.gBest - self.X[i])
                # # 速度取边界，位置归一化
                self.X[i] += self.V[i]
                self.X[i] = self.normalization(self.X[i])
                self.update(i, t)
            # # 把每一代的种群写入文件中
            # write_rows(self.Ovs, Ovs_p, None, 'a')
            time2 = time.time()
            total_t = time2-time1
            self.timeCost[t,:2] = [total_t, total_t - self.timeCost[t,-1]]
            self.check(t)
            # self.worst_ovs[t] = self.Ovs[np.argmin(self.Ovs[:, -1])]

        # network + community
    def iterator_netg(self, ver_network, g, c, Alpha, Beta, Gamma, vers, percen):
        # 时间开销：total time in one iteration of PSO, time_svm, pj_svm
        for t in range(1, self.Gm+1):
            # 更新velocity和position
            time3 = time.time()
            for i in range(self.N):
                self.r1, self.r2 = np.random.random(2)
                self.V[i] = self.w*self.V[i] + self.c1*self.r1*(self.pBest[i] - self.X[i])+ self.c2*self.r2*(self.gBest - self.X[i])
                # ll = list(range(self.dim))
                random.shuffle(ver_network)
                u_decode = self.indv_decode(self.X[i])
                for j in ver_network:
                    # 第j位对应的特征节点的加权度
                    djw = g.get_weight_degree(j)
                    # 该特征节点的邻接点中被置为1的加权个数，在变异后的个体中选择元素，并在graph中找该特征节点与对应特征节点的weight，并求和
                    wjn = g.cal_weight_number(j, u_decode)
                    if vers[j] != 2:
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
                self.X[i] += self.V[i]
                self.X[i] = self.normalization(self.X[i])
                self.update(i, t)
            # # 把每一代的种群写入文件中
            # write_rows(self.Ovs, Ovs_p, None, 'a')
            time4 = time.time()
            total_t = time4-time3
            self.timeCost[t,:2] = [total_t, total_t - self.timeCost[t,-1]]
            self.check(t)

    # 只有network
    def iterator_network(self, ver_network, g, Alpha, Beta, percen):
        # 时间开销：total time in one iteration of PSO, time_svm, pj_svm
        for t in range(1, self.Gm+1):
            # 更新velocity和position
            time3 = time.time()
            for i in range(self.N):
                self.r1, self.r2 = np.random.random(2)
                self.V[i] = self.w*self.V[i] + self.c1*self.r1*(self.pBest[i] - self.X[i])+ self.c2*self.r2*(self.gBest - self.X[i])
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
                self.X[i] += self.V[i]
                self.X[i] = self.normalization(self.X[i])
                self.update(i, t)
            # # 把每一代的种群写入文件中
            # write_rows(self.Ovs, Ovs_p, None, 'a')
            time4 = time.time()
            total_t = time4-time3
            self.timeCost[t,:2] = [total_t, total_t - self.timeCost[t,-1]]
            self.check(t)
    
    def update(self, i, t):
        # self.X[i] = self.normalization(self.X[i])
        self.Ovs[i] = self.obj_fun(self.X[i], t)
        if(self.Ovs[i, -1] > self.p_ovs[i][-1]):  # 更新个体最优
            self.p_ovs[i] = self.Ovs[i]
            self.pBest[i] = self.X[i]
            if(self.p_ovs[i][-1] > self.g_ovs[-1]): # 更新全局最优
                self.gBest = self.X[i].copy()
                self.g_ovs = self.p_ovs[i].copy()
    
    def check(self, t):
        self.metrics[t] = self.g_ovs.copy()
        print('iteration %d, ratio=%.4f, accuracy=%.4f, fitness=%.4f' % (t, self.metrics[t,-3], self.metrics[t,-2], self.metrics[t,-1])) # 输出每一代的最优解  

    def execution(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        g, c = None, None
        Alpha, Beta, Gamma = float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])
        path_network = sys.argv[3]
        path_com = sys.argv[4]
        flag = sys.argv[11] # 3代表NetG-PSO, 2代表Net-PSO, 1代表PSO
        percen = float(sys.argv[13])
        # path_network = '../../../data/processedData/credit-card-feat_network_norm_pearson06.txt'
        # path_com = '../../../data/processedData/credit-card-com0.8.txt'
        # path_network = '../../data/shipp-2002-v1-norm-pearson06.txt'
        # path_com = '../../data/shipp-norm-com0.75.txt'
        # Alpha, Beta, Gamma = 0.3, 0.3, 0.4
        # flag = '3'
        # percen = 0.2
        timestamp = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
        fit_h = ['Tpr', 'Fpr', 'Precission', 'Auc', 'Ratio', 'Accuracy', 'Fitness']
        if flag=='1':
            algorithm = 'PSO'
            print('rank %d running %s for %s' % (rank, algorithm, dataset))
            str1 = '{} {} weight_{} Iteration_{} N_{} Percen_{} w_{} rank_{} {}'.format(dataset, algorithm, self.weight, self.Gm, self.N, percen, self.w, rank, timestamp)
            init_time = self.initialization_random(initial_dir+'Initialization '+str1+'.csv', fit_h)
            self.iterator_standard()
        elif flag=='2':
            algorithm='Net-PSO'
            g = Graph1.Graph(path_network, self.weight)
            ver_network = list(g.getVertices())
            vers = np.zeros(self.dim, dtype=int)
            vers[ver_network] = 1
            ver_indp = np.where(vers==0)[0]  # 独立节点（不在网络结构中的节点）
            print('rank %d running %s for %s, weight=%.2f' % (rank, algorithm, dataset, self.weight))
            # Alpha, Beta = 0.5, 0.5
            str1 = '{} {} weight_{} Iteration_{} N_{} Percen_{} w_{} rank_{} {}'.format(dataset, algorithm, self.weight, self.Gm, self.N, percen, self.w, rank, timestamp)
            init_time = self.initialization_network(self.N, g, ver_network, ver_indp, Alpha, Beta, initial_dir+'Initialization '+str1+'.csv', fit_h)
            # Ovs_p = ovs_dir + 'Ovs ' + str1 + '.csv'
            self.iterator_network(ver_network, g, Alpha, Beta, percen)
        else:
            algorithm = 'NetG-PSO'
            g = Graph1.Graph(path_network, self.weight)
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
        fit_path = excels_dir + 'excels ' + str1 + '.csv'
        time_path = timeEstimation_dir + 'timeEstimation ' + str1 + '.csv'
        time_h = ['Total_t', 'Search_t', 'Svm_t']
        f = open(logpath+'log-PSO(10-3).txt', 'a')
        f.write(algorithm+' initialization cost '+str(init_time)+' seconds\n')
        f.close()
        write_rows(self.metrics, fit_path, fit_h, 'w')
        write_rows(self.timeCost, time_path, time_h, 'w')
        print('rank {} finished {} for {}'.format(rank, algorithm, dataset))

# %%
if __name__ == "__main__":
    np.random.seed(int(np.random.rand()*1000000))
    dataset=sys.argv[10]
    # dataset = 'Shipp'
    dim = trainData.shape[1]-1
    print('dim=', dim)
    # pN=200
    # Gm = 100
    pN = int(sys.argv[12])
    Gm=int(sys.argv[9])
    w = float(sys.argv[14])
    c1, c2 = 2, 2
    Pm = 0.01
    pso = PSO(pN, dim, Gm, w, c1, c2)
    pso.execution()

#%%
