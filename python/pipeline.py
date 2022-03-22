import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader

from itertools import chain

from nets import Synthesizer, Recognizer, Renderer, MakeFaces
from test import test
from train import train

from utils import COCOval


n = 32
m = 32

background_size = m * 2
resize_source_back = background_size * 2

batch_size = 16
#num_strings = 1000
epochs = 1
lr = 1e-4
weight_decay = 1e-5

synt_net_name = '../trained_nets/synt_net.pth'
rend_net_name = '../trained_nets/rend_net.pth'
rec_net_name = '../trained_nets/rec_net.pth'
gen_net_name = '../trained_nets/gen_net.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def training_pipeline(save=True):
	background_train = COCOval(resize_source_back,
               root_dir='../background/train', 
               transform = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.RandomCrop(background_size)
               ]))

	background_dataloader_train = torch.utils.data.DataLoader(background_train, 
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=2)
                                                    
	synt_net = Synthesizer(n, m)
	rend_net = Renderer(background_size)
	rec_net = Recognizer(n)
	gen_net = MakeFaces(m, device)
	
	synt_net.to(device)
	rend_net.to(device)
	rec_net.to(device)
	gen_net.to(device)
	
	criterion = nn.Sigmoid()
	
	params = chain(synt_net.parameters(), gen_net.parameters(), rend_net.parameters(), rec_net.parameters())
	optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
	
	print('Start training')
	train(device, background_dataloader_train, 
                synt_net, gen_net, rend_net, rec_net, 
                criterion, optimizer, epochs, n, m)
	print('Finish training')
        
	torch.save(gen_net.state_dict(), gen_net_name)
	if save:
                torch.save(synt_net.state_dict(), synt_net_name)
                torch.save(rend_net.state_dict(), rend_net_name)
                torch.save(rec_net.state_dict(), rec_net_name)
                torch.save(gen_net.state_dict(), gen_net_name)


def testing_pipeline():
	background_test = COCOval(resize_source_back,
               root_dir='../background/val', 
               transform = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize(background_size)
               ]))

	background_dataloader_test = torch.utils.data.DataLoader(background_test, 
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=2)

	synt_net = Synthesizer(n, m)
	rend_net = Renderer(background_size)
	rec_net = Recognizer(n)
	gen_net = MakeFaces(m, device)

	synt_net.load_state_dict(torch.load(synt_net_name))
	rend_net.load_state_dict(torch.load(rend_net_name))
	rec_net.load_state_dict(torch.load(rec_net_name))
	gen_net.load_state_dict(torch.load(gen_net_name))
	
	synt_net.to(device)
	rend_net.to(device)
	rec_net.to(device)
	gen_net.to(device)
        
	synt_net.eval()
	rend_net.eval()
	rec_net.eval()
	gen_net.eval()

	print('Start testing')
	test(device, background_dataloader_test, 
                synt_net, gen_net, rend_net, rec_net, n, m)

if __name__ == '__main__':
        training_pipeline(False)
        testing_pipeline()

