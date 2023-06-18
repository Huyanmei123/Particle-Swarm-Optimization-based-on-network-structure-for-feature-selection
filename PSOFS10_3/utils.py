import os
data_dir = './data/'
result_dir = './result4/'
initial_dir = result_dir+'initialization/'
excels_dir = result_dir+'excels/'
timeEstimation_dir = result_dir+'timeEstimation/'

result_d_dir = './result_Delta/'
initial_d_dir = result_d_dir+'initialization/'
excels_d_dir = result_d_dir+'excels/'
timeEstimation_d_dir = result_d_dir+'timeEstimation/'

result_N_dir = './result_N/'
initial_N_dir = result_N_dir+'initialization/'
excels_N_dir = result_N_dir+'excels/'
timeEstimation_N_dir = result_N_dir+'timeEstimation/'

result_percen_dir = './result_Percen/'
initial_percen_dir = result_percen_dir+'initialization/'
excels_percen_dir = result_percen_dir+'excels/'
timeEstimation_percen_dir = result_percen_dir+'timeEstimation/'

result_w_dir = './result_w/'
initial_w_dir = result_w_dir+'initialization/'
excels_w_dir = result_w_dir+'excels/'
timeEstimation_w_dir = result_w_dir+'timeEstimation/'

log_dir = './log/'
# dirs = [result_dir, initial_dir, excels_dir, timeEstimation_dir, result_d_dir, initial_d_dir, excels_d_dir, timeEstimation_d_dir, result_N_dir, initial_N_dir, excels_N_dir, timeEstimation_N_dir, result_percen_dir, initial_percen_dir, excels_percen_dir, timeEstimation_percen_dir, result_w_dir, initial_w_dir, excels_w_dir, timeEstimation_w_dir, log_dir]
dirs = [result_dir, initial_dir, excels_dir, timeEstimation_dir, log_dir]

for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('create ', dir)


