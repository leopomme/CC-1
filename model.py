import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessPieceCNN(nn.Module):
    def __init__(self, num_classes=13):  # Adjust the number of classes as needed
        super(ChessPieceCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 32 * 32, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x