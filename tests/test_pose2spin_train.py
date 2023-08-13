from pathlib import Path
import numpy as np
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

class CustomDataset(Dataset):
    def __init__(self,dataset_path,seq_size= 8):
        
        self.poses = None
        self.spins = None

        right_arm = [10,8,6]


        p = Path(dataset_path)
        oup_paths = list(p.glob('**/d_spin_priors.csv'))
        inp_paths = list(p.glob('**/d_human_poses.csv'))

        for pin,pout in zip(inp_paths,oup_paths):
            oup_data = read_csv(pout)
            inp_data = read_csv(pin).reshape(len(oup_data),seq_size,26,2) # traj num, seq size, key pts, uv
            inp_data = inp_data - inp_data[:,:,19,None,:]
            inp_data = inp_data[:,:,right_arm,:]

            if self.poses is None:
                self.poses = inp_data
                self.spins = oup_data
            else:
                self.poses = np.concatenate((self.poses,inp_data),axis=0)
                self.spins = np.concatenate((self.spins,oup_data),axis=0)
        
        self.poses = torch.from_numpy(self.poses).float().to(device)
        self.spins = torch.from_numpy(self.spins).float().to(device)

    def __len__(self):
        return len(self.spins)

    def __getitem__(self, idx):
        return self.poses[idx,:], self.spins[idx,:]

    

    
    
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

def run():
    torch.manual_seed(0)
    dataset = CustomDataset('dataset',seq_size=100)

    training_data = Subset(dataset, [i for i in range(len(dataset)) if i % 2 == 1] )
    val_dataset = Subset(dataset, [i for i in range(len(dataset)) if i % 2 == 0] )
    # training_data, val_dataset = random_split(dataset,[0.75,0.25])
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)


    model = TCN().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-7,momentum=0.99,weight_decay=5e-5)

    epochs = 200
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == '__main__':
   run()
