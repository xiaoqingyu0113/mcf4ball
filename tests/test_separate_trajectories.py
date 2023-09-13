
import numpy as np
import os
import glob

from test_helper import read_csv,write_rows_to_csv,result_parser,plot_separated_results,plot_whole_results



def seperate_traj_by_gid(gid):
    prev_node_id = gid[0]
    result_ind = []
    for i, node_id in enumerate(gid):
        if node_id < prev_node_id:
            result_ind.append(i)
        prev_node_id = node_id
    return result_ind

def save_separate_idx(folder_name):
    '''
    0. need to execute "run" first to get the results
    1. compute the trajectory ranges (ind_start, ind_end, iter_start, iter_end)
    2. save in d_separate_id.csv
    '''
    saved_p,saved_v,saved_w,saved_w0,saved_iter,saved_gid,saved_time,saved_detection_id = result_parser(folder_name+'/d_results.csv')
    traj_seperator = seperate_traj_by_gid(saved_gid)
    save_indices= []
    for i in range(len(traj_seperator)-1):
        s = traj_seperator[i]
        e = traj_seperator[i+1] -5
        dist_x = np.abs(saved_p[s,0] - saved_p[e,0])
        if  dist_x > 11.0 and saved_p[e,0]>-2 and saved_p[e,0] < saved_p[s,0]:
            save_indices.append([s,e,saved_iter[s],saved_iter[e],saved_detection_id[s],saved_detection_id[e]])
            # print('-----')
            # print(saved_p[s,:])
            # print(saved_p[e,:])
    write_rows_to_csv(folder_name+'/d_separate_ind.csv',save_indices)
 
 
def run_separate():
    # run all folder
    folders = glob.glob('dataset/tennis_*')
    for folder_name in folders:
        print('processing ' + folder_name)
        save_separate_idx(folder_name)
        plot_separated_results(folder_name,save_fig=True)


if __name__ == '__main__':
    # run_separate()
    folder_name = 'dataset/tennis_2'
    save_separate_idx(folder_name)
    plot_separated_results(folder_name,save_fig=False)
    # plot_whole_results('dataset/tennis_13')
    

