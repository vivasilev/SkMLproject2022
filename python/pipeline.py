import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain

from nets import Synthesizer, Recognizer
from test import test
from train import train

bit_string_length = 50
marker_size = 32

batch_size = 16
num_strings = 1000

lr = 1e-4
weight_decay = 1e-5
epochs = 400

synt_net_name = '../trained_nets/synt_net.pth'
rec_net_name = '../trained_nets/rec_net.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def training_pipeline(save=True):
        synt_net = Synthesizer(bit_string_length, marker_size)
        rec_net = Recognizer(bit_string_length)

        synt_net.to(device)
        rec_net.to(device)
	
        criterion = nn.Sigmoid()

        params = chain(synt_net.parameters(), rec_net.parameters())
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

        print('Start training')
        train(device, num_strings, batch_size, 
                synt_net, rec_net, 
		criterion, optimizer, epochs, bit_string_length)
        print('Finish training')

        if save:
                torch.save(synt_net.state_dict(), synt_net_name)
                torch.save(rec_net.state_dict(), rec_net_name)


def testing_pipeline():
        synt_net = Synthesizer(bit_string_length, marker_size)
        rec_net = Recognizer(bit_string_length)

        synt_net.load_state_dict(torch.load(synt_net_name))
        rec_net.load_state_dict(torch.load(rec_net_name))
	
        synt_net.to(device)
        rec_net.to(device)
	
        synt_net.eval()
        rec_net.eval()

        print('Start testing')
        test(device, num_strings, batch_size, 
		synt_net, rec_net, bit_string_length)

if __name__ == '__main__':
        training_pipeline()
        testing_pipeline()

