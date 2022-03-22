import torch
import torch.nn as nn
import torch.optim as optim
import random

from tqdm import tqdm

from dataset import get_celeba
from architecture import weights_init, Generator, Discriminator
from dpipe.io import load

# Set random seed for reproducibility.
seed = 42
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

params = load('params.json')

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

dataloader = get_celeba(params)

net_generator = Generator(params).to(device).apply(weights_init)
net_discriminator = Discriminator(params).to(device).apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

real_label = 1
fake_label = 0

optimizer_generator = optim.Adam(net_generator.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optimizer_discriminator = optim.Adam(net_discriminator.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Stores generated images as training progresses.
img_list = []

generator_losses, discriminator_losses = [], []

for epoch in tqdm(range(params['nepochs'])):
    for i, data in enumerate(dataloader, 0):

        real_data = data[0].to(device)
        b_s = real_data.size(0)

        net_discriminator.zero_grad()

        label = torch.full((b_s,), real_label, device=device).float()
        output = net_discriminator(real_data).view(-1)
        errD_real = criterion(output, label)

        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_s, params['nz'], 1, 1, device=device)

        fake_data = net_generator(noise)

        label.fill_(fake_label)

        output = net_discriminator(fake_data.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizer_discriminator.step()

        net_generator.zero_grad()

        label.fill_(real_label)

        output = net_discriminator(fake_data).view(-1)
        errG = criterion(output, label)

        errG.backward()

        D_G_z2 = output.mean().item()

        optimizer_generator.step()

        # Check progress of training.
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        generator_losses.append(errG.item())
        discriminator_losses.append(errD.item())

    # Save the model.
    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator': net_generator.state_dict(),
            'discriminator': net_discriminator.state_dict(),
            'optimizerG': optimizer_generator.state_dict(),
            'optimizerD': optimizer_discriminator.state_dict(),
            'params': params
    }, 'checkpoints/model_epoch_{}.pth'.format(epoch), )

# Save the final trained model.
torch.save({
    'generator': net_generator.state_dict(),
    'discriminator': net_discriminator.state_dict(),
    'optimizerG': optimizer_generator.state_dict(),
    'optimizerD': optimizer_discriminator.state_dict(),
    'params': params
}, 'model/model_final.pth')
