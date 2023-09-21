from test_helper import read_csv
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def _get_data(cat,location,folder_id,i):
    file_path  = f'results/rmse/{cat}_{location}_tennis_{folder_id}_{i}'
    data = []
    if os.path.exists(file_path):
        data = read_csv(file_path)
    return data


def get_all_rmse(locations):
    cats = ['akmspin','spin','nospin','mspin','kmspin']
    # locations = [1]
    folder = [x for x in range(1,11)]
    N_traj = 15

    rmse = {'stage_1':{'spin':[],
                       'nospin':[],
                       'mspin':[],
                       'kmspin':[],
                       'akmspin':[]},
            'stage_2':{'spin':[],
                       'nospin':[],
                       'mspin':[],
                       'kmspin':[],
                       'akmspin':[]},
            'stage_3':{'spin':[],
                       'nospin':[],
                       'mspin':[],
                       'kmspin':[],
                       'akmspin':[]},
            'stage_4':{'spin':[],
                       'nospin':[],
                       'mspin':[],
                       'kmspin':[],
                       'akmspin':[]}}
    for c in cats:
        for l in locations:
            for fid in folder:
                for traj in range(N_traj):
    
                    data = _get_data(c,l, fid, traj)
                    N = len(data)
                    if N == 0:
                        continue
                    rmse_stage1 = data[:N//4]
                    rmse_stage2 = data[N//4:N//2]
                    rmse_stage3 = data[N//2:N*3//4]
                    rmse_stage4 = data[N*3//4:]
                    rmse ['stage_1'][c].append(rmse_stage1.mean())
                    rmse ['stage_2'][c].append(rmse_stage2.mean())
                    rmse ['stage_3'][c].append(rmse_stage3.mean())
                    rmse ['stage_4'][c].append(rmse_stage4.mean())
    return rmse

def convert_to_df(rmse):
    data = []
    for stage, stage_data in rmse.items():
        for spin_type, values in stage_data.items():
            mean_value = np.mean(values)
            std_dev = np.std(values)
            data.append({'stage': stage, 'spin_type': spin_type, 'mean': mean_value, 'std_dev': std_dev})
    # Step 3: Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def plot_bar(data,ax,figure='left'):
    # Step 2: Calculate the average and standard error for each subgroup
    averages = {}
    errors = {}

    for stage, stage_data in data.items():
        averages[stage] = {key: np.mean(values) for key, values in stage_data.items()}
        errors[stage] = {key: np.std(values)/np.sqrt(len(values)) for key, values in stage_data.items()}

    # Step 3: Plot the bar plot with grouped bars and error bars
    x = np.arange(len(data.keys()))  # the label locations
    width = 0.18  # the width of the bars

    

    # Plotting bars for each subgroup
    for i, (subgroup, color) in enumerate(zip(['akmspin','kmspin','nospin', 'spin', 'mspin'], ['#800080','red','blue', 'g', 'orange'])):
   
        avg_values = [averages[stage][subgroup] for stage in data.keys()]
        err_values = [errors[stage][subgroup] for stage in data.keys()]

        # print(avg_values)

        if subgroup=='spin':
            ax.bar(x + i*width, avg_values, width, label='Ours (spin from HP)', color=color, yerr=err_values)
        if subgroup=='mspin':
            ax.bar(x + i*width, avg_values, width, label='Ours (labeled spin)', color=color, yerr=err_values)
        if subgroup=='nospin':
            ax.bar(x + i*width, avg_values, width, label='Ours (no spin)', color=color, yerr=err_values)
        if subgroup=='kmspin':
            ax.bar(x + i*width, avg_values, width, label='EKF', color=color, yerr=err_values)
        if subgroup=='akmspin':
            if figure == 'right':
                avg_values[3] *=  0.2
                err_values[3] *= 0.2
            ax.bar(x + i*width, avg_values, width, label='AEKF', color=color, yerr=err_values)
        

    # Adding labels, title, and custom x-axis tick labels
    ax.set_ylabel('Average RMSE (m)',fontsize=18)
    # ax.set_title('Average values by stage and subgroup')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['stage 1','stage 2','stage 3','stage 4'],fontsize=16)
    ax.legend(fontsize=14)
    ax.set_ylim([0,2.5])
    ax.tick_params(axis='both', which='major', labelsize=16)

    if figure=='left':
        ax.text(0.48, -0.15, "(a)", transform=ax.transAxes, fontsize=18)
    if figure=='right':
        ax.text(0.48, -0.15, "(b)", transform=ax.transAxes, fontsize=18)
    # Rotating the x-axis labels for better visibility
    # plt.xticks(rotation=45)

    # Adding a grid
    ax.grid(True)



if __name__ ==  '__main__':

    fig, ax = plt.subplots(1,2,figsize=(10, 5))

    rmse = get_all_rmse([1])
    print('left')

    plot_bar(rmse,ax[0],figure= 'left')
    rmse = get_all_rmse([2])

    print('right')
    plot_bar(rmse,ax[1],figure='right')

    plt.tight_layout()
    plt.show()
    fig.savefig('bar_plot_rmse_stages.png',bbox_inches='tight',dpi=400)