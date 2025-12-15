import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalCNN(nn.Module):
    def __init__(self):
        super(MedicalCNN, self).__init__()
        # Input: 1 channel (Grayscale), 28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout(0.5)
        # 28x28 -> pool -> 14x14. 64 channels. 
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # Double pooling for small MNIST
        # Standardize shape if dimensions mismatch (resilience)
        if x.shape[-1] != 7: 
            x = F.adaptive_avg_pool2d(x, (14, 14))
            
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x