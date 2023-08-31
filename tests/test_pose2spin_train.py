from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader, Subset, random_split
from torch import nn
from mcf4ball.pose2spin import *


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def read_csv(path):
    with open(path,'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = np.array(list(reader),dtype=float)
    return data
def print_dict(d):
    for k,v in d.items():
        print('==============')
        print(k,':')
        for kk,vv in v.items():
            print('\t',kk,' : ',vv.shape)
class CustomDataset(Dataset):
    def __init__(self,dataset_path,max_seq_size= 100,seq_size = 20):
        
        self.max_seq_size = 100
        self.step = int(np.floor(max_seq_size/seq_size))

        self.poses = None
        self.spins = None

        right_arm = [10,8,6]


        p = Path(dataset_path)
        oup_paths = list(p.glob('**/d_spin_priors.csv'))
        inp_paths = list(p.glob('**/d_human_poses.csv'))
        iter_indices = list(p.glob('**/d_sperate_id.csv'))
    

        self.dataset_dict = dict()
        # oup_paths = [p for p in oup_paths if ('3' not in str(p)) or ('4' not in str(p))] 
        # inp_paths = [p for p in inp_paths if ('3' not in str(p)) or ('4' not in str(p))] 

        for pin,pout, piter in zip(inp_paths,oup_paths,iter_indices):
            folder_name = str(pin).split('/')[1]
            oup_data = read_csv(pout)
            inp_data = read_csv(pin).reshape(len(oup_data),max_seq_size,26,2) # traj num, seq size, key pts, uv
            iters = read_csv(piter)[:,2:4]
            # print(len(oup_data))
            # print(len(oup_data))
            # print(len(iters))
            inp_data = inp_data - inp_data[:,:,19,None,:]
            inp_data = inp_data[:,:,right_arm,:]

            inp_temp = []
            oup_temp = []
            iters_temp = []
            for inp,oup,iter in zip(inp_data,oup_data,iters):
                if np.linalg.norm(oup) < 3.0:
                    continue
                inp_temp.append(inp)
                oup_temp.append(oup)
                iters_temp.append(iter)
            inp_data = np.array(inp_temp)
            oup_data = np.array(oup_temp)
            iters = np.array(iters_temp)
            
            self.dataset_dict[folder_name] = {'iters':iters,
                                        'poses':inp_data,
                                        'labels':oup_data} 

            if self.poses is None:
                self.poses = inp_data
                self.spins = oup_data
            else:
                self.poses = np.concatenate((self.poses,inp_data),axis=0)
                self.spins = np.concatenate((self.spins,oup_data),axis=0)
        
        self.poses = torch.from_numpy(self.poses).float().to(device)
        self.spins = torch.from_numpy(self.spins).float().to(device)
        print_dict(self.dataset_dict)

    def __len__(self):
        return len(self.spins)

    def __getitem__(self, idx):
        s = int(np.floor(np.random.rand()*5))
        rg = list(range(s,self.max_seq_size,self.step))
        return self.poses[idx,rg,:], self.spins[idx,:]

    

    
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # loss, current = loss.item(), (batch + 1) * len(X)
        # if batch  % 10 ==0:
    print(f"Traning loss: {loss:>7f}")
    return loss

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
        # print(pred)
        # print(y)
    test_loss /= num_batches
    print(f"Validation loss: {test_loss} \n")
    return test_loss

def run():
    torch.manual_seed(0)
    dataset = CustomDataset('dataset',max_seq_size=100,seq_size=20)

    training_data = Subset(dataset, [i for i in range(len(dataset)) if i % 2 == 1] )
    val_dataset = Subset(dataset, [i for i in range(len(dataset)) if i % 2 == 0] )
    # training_data, val_dataset = random_split(dataset,[0.8,0.2])
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)


    model = TCN().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7,momentum=0.99,weight_decay=5e-5)

    epochs = 500
    training_loss = []
    testing_loss = []
    min_loss_test = 999999
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss_test  = test_loop(test_dataloader, model, loss_fn)
        loss_train = train_loop(train_dataloader, model, loss_fn, optimizer)
        training_loss.append(loss_train.cpu().item())
        testing_loss.append(loss_test)
        if loss_test < min_loss_test:
            torch.save(model, 'trained/tcnn.pth')

    print("Done!")

    training_rmse = np.sqrt(np.array(training_loss))/(np.pi*2)
    testing_rmse = np.sqrt(np.array(testing_loss))/(np.pi*2)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(training_rmse,'b',label = 'train')
    ax.plot(testing_rmse,'r',label='test')
    ax.set_title(min(testing_rmse))
    ax.legend()
    plt.show()

def show_prediction():
    torch.manual_seed(0)
    dataset = CustomDataset('dataset',max_seq_size=100,seq_size=20)
    val_dataset = Subset(dataset, [i for i in range(len(dataset)) if i % 2 == 0] )
    model = torch.load('trained/tcnn.pth').to(device)
    model.eval()
    with torch.no_grad():
        for inp,label in val_dataset:
            pred = model(inp[None,:])
            print('=========================================================')
            print('label \t\t=\t',label/6.28)
            print('prediction \t=\t', pred/6.28)

if __name__ == '__main__':
    # run()
    show_prediction()