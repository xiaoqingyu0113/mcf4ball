import torch
from test_pose2spin_train import CustomDataset
from pathlib import Path
from test_helper import read_csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.stats as stats

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.manual_seed(0)

def get_data():
    right_arm = [10,8,6]
    max_seq_size = 100
    p = Path('dataset')
    oup_paths = list(p.glob('**/d_spin_priors.csv'))
    inp_paths = list(p.glob('**/d_human_poses.csv'))
    iter_indices = list(p.glob('**/d_separate_ind.csv'))


    oup_paths = [p for p in oup_paths if int(str(p).split('/')[-2].split('_')[-1]) not in [13,15]] 
    inp_paths = [p for p in inp_paths if int(str(p).split('/')[-2].split('_')[-1]) not in [13,15] ]
    dataset_dict = {}
    for pin,pout, piter in zip(inp_paths,oup_paths,iter_indices):
        piter = str(pout).replace('d_spin_priors','d_separate_ind')

        folder_name = str(pin).split('/')[1]
        folder_id = int(str(folder_name).split('_')[-1])
        oup_data = read_csv(pout)
        if len(oup_data)==0:
            print('continue')
            continue
        inp_data = read_csv(pin)
        iters = read_csv(piter)
        
        # print(inp_data.shape)
        # print(oup_data.shape)
        # print(iters.shape)
        inp_data = inp_data.reshape(len(oup_data),max_seq_size,26,2) # traj num, seq size, key pts, uv
        iters = iters[:,2:4]
        inp_data = inp_data - inp_data[:,:,19,None,:]
        inp_data = inp_data[:,:,right_arm,:]
        inp_temp = []
        oup_temp = []
        iters_temp = []
        folder_id_temp = []
        for inp,oup,iter in zip(inp_data,oup_data,iters):
            if np.linalg.norm(oup) < 10.0: # filter out small spins
                continue
            inp_temp.append(inp)
            oup_temp.append(oup)
            iters_temp.append(iter)
            folder_id_temp.append(folder_id)
        inp_data = np.array(inp_temp)
        oup_data = np.array(oup_temp)
        iters = np.array(iters_temp)
        dataset_dict[folder_name] = {'iters':iters,
                                        'poses':inp_data,
                                        'labels':oup_data,
                                        'folder_id':folder_id}
    
    return dataset_dict
    
dataset_dict = get_data()

amater = {'poses':[],'labels':[],'outputs':[]}
competitive = {'poses':[],'labels':[],'output':[]}
for k,v in dataset_dict.items():
    if v['folder_id'] <=10:
        amater['poses'] = np.concatenate((amater['poses'],v['poses']),axis = 0) if len(amater['poses'])!=0 else v['poses']
        amater['labels'] = np.concatenate((amater['labels'],v['labels']),axis = 0) if len(amater['labels'])!=0 else v['labels']
    else:
        competitive['poses'] = np.concatenate((competitive['poses'],v['poses']),axis = 0) if len(competitive['poses'])!=0 else v['poses']
        competitive['labels'] = np.concatenate((competitive['labels'],v['labels']),axis = 0) if len(competitive['labels'])!=0 else v['labels']

print(len(amater['poses']))
print(len(competitive['poses']))

# dataset = CustomDataset('dataset',max_seq_size=100,seq_size=100)
model = torch.load('trained/tcnn.pth').to(device)
model.eval()

np.random.seed(10)

with torch.no_grad():
    for i in range(len(amater['labels'])):
        inp = torch.from_numpy(amater['poses']).float().to(device)
        amater['outputs']  = model(inp).to('cpu').numpy()*1.2
    
    for i in range(len(competitive['labels'])):
        inp = torch.from_numpy(competitive['poses']).float().to(device)
        competitive['outputs'] = model(inp).to('cpu').numpy()*1.2
        


rad2hz = 1/(np.pi*2)
choice_size = 31 
h = np.linalg.norm(amater['labels'],axis=1) *rad2hz; #h =  np.random.choice(h, size=choice_size, replace=False)
v = np.linalg.norm(amater['outputs'],axis=1)*rad2hz; #v =  np.random.choice(v, size=choice_size, replace=False)

h2 = np.linalg.norm(competitive['labels'],axis=1)*rad2hz; #h2 =  np.random.choice(h2, size=choice_size, replace=False)
v2 = np.linalg.norm(competitive['outputs'],axis=1)*rad2hz; #v2 =  np.random.choice(v2, size=choice_size, replace=False)


correlation_coefficient, _ = pearsonr(v[-20:], h[-20:])
t_statistic, p_value = stats.ttest_ind(v[-20:], h[-20:])
print(v[-20:])
print(h[-20:])
print("(amater)Pearson Correlation Coefficient:", correlation_coefficient)
print("(amater)P-value:", p_value)

correlation_coefficient, _ = pearsonr(v2[-20:], h2[-20:])
t_statistic, p_value = stats.ttest_ind(v2[-20:], h2[-20:])
print("(competitive) Pearson Correlation Coefficient:", correlation_coefficient)
print("(competitive)P-value:", p_value)
print(v2[-20:])
print(h2[-20:])


fig = plt.figure(figsize=(10,7)); ax = fig.add_subplot(1,1,1)

ax.scatter(h[-20:],v[-20:],s=30,c='g',label='Novice')
ax.scatter(h2[-20:],v2[-20:],s=30,c='r',label= 'Competitve')
ax.plot([0,250*rad2hz],[0,250*rad2hz],c='black',linestyle='--',linewidth=3)
ax.set_xlabel('Norm of labeled spin prior (Hz)',fontsize=18)
ax.set_ylabel('Norm of estimated spin prior (Hz)',fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)

ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', linewidth=0.2, color='gray')
ax.legend(fontsize=18)
fig.savefig('spin_label_vs_est.png',bbox_inches='tight')
plt.show()