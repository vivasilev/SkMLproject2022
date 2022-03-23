import torch
import torch.nn as nn

import numpy as np
import kornia as K

from utils import Reshape
from dcgan.inference import predict

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


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
        #self.fc = nn.Linear(192, n) # w/o background
        self.fc = nn.Linear(576, n) # with background
        
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
    
    
class Renderer(nn.Module):
    def __init__(self, size):
        super(Renderer, self).__init__()        
        self.color_jitter = K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5)
        self.size = size # size of marker (with background)
  
    def forward(self, x):
        # affine transformation
        affine_sigma = np.random.uniform(0.01, 0.1, 1)
        affine_matrix = torch.tensor([[1, 0, 0], [0, 1, 0]]) +\
            torch.normal(torch.zeros(2, 3), torch.full((2, 3), float(affine_sigma)))
        x = K.geometry.transform.affine(x.float(), affine_matrix)
        
        # noise
        for i in range(x.shape[0]):
            # same parameters for each channel
            g = float(np.random.uniform(0.001, 0.003, 1))
            Jahne_sigma = float(np.random.uniform(0.001, 0.3, 1))
            
            # print(torch.mean(x[i], axis=[1, 2]))
            std = torch.sqrt(g * torch.mean(x[i], axis=[1, 2]) + Jahne_sigma ** 2)
            for j in range(x[i].shape[0]):
                std_j = 0.1 if torch.isnan(std[j]) else float(std[j])
                x[i][j] = x[i][j] + torch.normal(
                    torch.zeros(self.size, self.size), torch.full((self.size, self.size), std_j)
                )
                # x[i][j] = torch.clip(x[i][j], 0, 1)
        
        # color transformation
        x = self.color_jitter(x)
        
        # blurring
        blur_sigma = np.random.uniform(0.001, 0.5, 1)
        x = K.filters.gaussian_blur2d(x, (3, 3), (blur_sigma, blur_sigma))
        return x


class MakeFaces(nn.Module):
    def __init__(self, m, device):
        super(MakeFaces, self).__init__()
        self.reshape = Reshape(-1, 3072)
        self.linear = nn.Linear(3072, 100)
        self.device = device
        self.m = m
        
    def forward(self, x):
        x = self.reshape(x)
        x = self.linear(x.float())
        
        x = predict(x, device=self.device, 
              model_path='../trained_nets/generator.pth', 
              params_path='../python/dcgan/params.json')
        
        x = K.geometry.transform.scale(x,
            scale_factor=torch.full((x.shape[0], 1), 0.5))
        x = x[:, :, (self.m // 2):(self.m + self.m//2), (self.m // 2):(self.m + self.m//2)]
        return x
