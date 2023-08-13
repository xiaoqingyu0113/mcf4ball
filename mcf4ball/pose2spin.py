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
        self.ln1 = nn.LayerNorm([16, 100])
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.ln2 = nn.LayerNorm([64, 50])
        
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        
        # Fully Connected layers
        self.fc1 = nn.Linear(128 * 12, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # Reshape data for 1D convolution
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), -1, x.size(1))
        
        # Apply Conv1D layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = self.ln1(x)
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.ln2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
class TemporalCNN(nn.Module):
    def __init__(self):
        super(TemporalCNN, self).__init__()
        
        # Temporal Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv1d(in_channels=3*2, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Fully Connected layers with Layer Normalization and Dropout
        self.fc1 = nn.Linear(256 * 100, 512)
        self.ln1 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        # Reshape data to combine human key points and pixel coordinates for 1D convolution (N,kps,L=100)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), -1, x.size(1))
        
        # Apply temporal convolutions with Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with Layer Normalization and Dropout
        x = self.dropout1(F.relu(self.ln1(self.fc1(x))))
        x = self.dropout2(F.relu(self.ln2(self.fc2(x))))
        x = self.fc3(x)
        
        return x