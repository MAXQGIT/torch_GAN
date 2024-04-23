import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
img_shape = (1, 28, 28)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

z = torch.FloatTensor(np.random.normal(0, 1, (64, 100))).to(device)
generator = torch.load('generator.pt')
gen_imgs = generator(z)
save_image(gen_imgs.data[:25], "images/d.png" , nrow=5, normalize=True)