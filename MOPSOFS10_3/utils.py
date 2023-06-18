import os
data_dir = './data/'
result_dir = './result/'
initial_dir = result_dir+'initialization/'
Dom_dir = result_dir+'Dom/'
Gbest_dir = result_dir + 'Gbest/'
timeEstimation_dir = result_dir+'timeEstimation/'

# result_w_dir = './result_w/'
# initial_w_dir = result_w_dir+'initialization/'
# Dom_w_dir = result_w_dir+'Dom/'
# Gbest_w_dir = result_w_dir + 'Gbest/'
# timeEstimation_w_dir = result_w_dir+'timeEstimation/'

# result_N_dir = './result_N/'
# initial_N_dir = result_N_dir+'initialization/'
# Dom_N_dir = result_N_dir+'Dom/'
# Gbest_N_dir = result_N_dir + 'Gbest/'
# timeEstimation_N_dir = result_N_dir+'timeEstimation/'

# result_percen_dir = './result_Percen/'
# initial_percen_dir = result_percen_dir+'initialization/'
# Dom_percen_dir = result_percen_dir+'Dom/'
# Gbest_percen_dir = result_percen_dir + 'Gbest/'
# timeEstimation_percen_dir = result_percen_dir+'timeEstimation/'

# result_d_dir = './result_Delta/'
# initial_d_dir = result_d_dir+'initialization/'
# Dom_d_dir = result_d_dir+'Dom/'
# Gbest_d_dir = result_d_dir + 'Gbest/'
# timeEstimation_d_dir = result_d_dir+'timeEstimation/'

log_dir = './log/'
dirs = [result_dir, initial_dir, Dom_dir, Gbest_dir, timeEstimation_dir]
# dirs = [result_dir, initial_dir, Dom_dir, Gbest_dir, timeEstimation_dir, result_percen_dir, initial_percen_dir, Dom_percen_dir, Gbest_percen_dir, timeEstimation_percen_dir, result_d_dir, initial_d_dir, Dom_d_dir, Gbest_d_dir, timeEstimation_d_dir, result_N_dir, initial_N_dir, Dom_N_dir, Gbest_N_dir, timeEstimation_N_dir, result_w_dir, initial_w_dir, Dom_w_dir, Gbest_w_dir, timeEstimation_w_dir, log_dir]

for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('create ', dir)

