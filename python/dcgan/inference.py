import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from dpipe.io import load

from .architecture import Generator


def predict(n, device, model_path, params_path, nz=100):
    params = load(params_path)
    net = Generator(params).to(device)
    net.load_state_dict(torch.load(model_path))

    noise = torch.randn(n, nz, 1, 1, device=device)

    with torch.no_grad():
        img = net(noise).detach().cpu()

    return img


def show_images(img):
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(make_grid(img, padding=2, normalize=True), (1, 2, 0)))
    plt.show()
