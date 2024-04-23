import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
img_shape = (1, 32, 32)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(10, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img
decoder = torch.load('decoder.pt')


def sample_image(n_row):
    """Saves a grid of generated digits"""
    # Sample noise
    z = torch.FloatTensor(np.random.normal(0, 1, (n_row ** 2, 10))).to(device)
    gen_imgs = decoder(z).to(device)
    save_image(gen_imgs.data, "images/d.png" , nrow=n_row, normalize=True)

sample_image(n_row=10)