import os
import torch
from torch import nn
from torch.utils.data import DataLoader

import torch.nn.functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        
        # Conv1D Layers
        self.conv1 = nn.Conv1d(in_channels=3*2, out_channels=16, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([16, 20])
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.ln2 = nn.LayerNorm([64, 10])
        
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        
        # Fully Connected layers
        self.myconv1 = nn.Conv1d(in_channels=3*2, out_channels=32, kernel_size=7, padding=3)
        self.myconv2= nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(1920, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,3)
        self.fc4 = nn.Linear(3,3)
    def forward(self,x):
        # Reshape data for 1D convolution
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), -1, x.size(1))
        x = self.myconv1(x)
        y= self.myconv2(x) 
        z = self.myconv2(y)

        x = x.view(x.size(0), -1)
        y = y.view(y.size(0),-1)
        z = z.view(y.size(0),-1)
        x = torch.cat((x,y,z),dim=1)
        # print(x.size())
        # print(y.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    # def forward(self, x):
    #     # Reshape data for 1D convolution
    #     x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), -1, x.size(1))
        
    #     # Apply Conv1D layers with ReLU activations
    #     x = F.relu(self.conv1(x))
    #     # x = self.ln1(x)
        
    #     x = F.relu(self.conv2(x))
    #     x = F.relu(self.conv3(x))
    #     x = F.relu(self.conv4(x))
    #     # x = self.ln2(x)
        
    #     # x = F.relu(self.conv5(x))
    #     # x = F.relu(self.conv6(x))
        
    #     # Flatten the tensor
    #     x = x.view(x.size(0), -1)
    #     # print(x.size())
    #     # Fully connected layers
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
        
    #     return x
    
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        
        self.fc1 = nn.Linear(120, 5012)
        self.fc2 = nn.Linear(5012, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 3)

    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
