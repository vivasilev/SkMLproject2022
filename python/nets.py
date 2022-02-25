import torch
import torch.nn as nn

from utils import Reshape


class Synthesizer(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.fc = nn.Linear(n, 3 * m ** 2)
        self.sigmoid = nn.Sigmoid()
        self.reshape = Reshape(-1, 3, m, m)
        self.conv = nn.Conv2d(3, 3, 3)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = x.float()
        x = self.sigmoid(self.fc(x))
        x = self.reshape(x)
        
        #x = self.upsample(x)
        
        # 16
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        #x = self.sigmoid(self.conv(x))
        
        return x.double()
    
    
class Recognizer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 96, 5, padding=2)
        self.conv = nn.Conv2d(96, 96, 5, padding=2)
        self.pool = nn.MaxPool2d(3, padding=(1, 0))
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(96)
        self.fc = nn.Linear(192, n)
        
    def forward(self, x):
        x = x.float()
        
        #x = self.batchnorm(self.relu(self.pool(self.conv0(x))))
        x = self.relu(self.pool(self.conv0(x)))
        #x = self.batchnorm(self.relu(self.pool(self.conv(x))))
        x = self.relu(self.pool(self.conv(x)))
        #x = self.batchnorm(self.relu(self.pool(self.conv(x))))
        x = self.relu(self.pool(self.conv(x)))
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x.double()

