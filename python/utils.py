import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from skimage import io, transform


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class COCOval(Dataset):
    '''
    MS COCO validation images without annotation
    '''
    def __init__(self, size, root_dir, transform=None):
        '''
        Args:
            size (int or tuple): Square scaling size for all images
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        self.root_dir = root_dir
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                os.listdir(self.root_dir)[idx])
        image = io.imread(img_name)
        image = transform.resize(image, (self.size, self.size))
        
        if self.transform:
            image = self.transform(image)

        return image

