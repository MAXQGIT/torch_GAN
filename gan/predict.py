import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt

img_shape = (1, 28, 28)


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
        img = img.view(img.size(0), *img_shape)
        return img


device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = torch.load('gan.pt')
print(generator)
z = torch.tensor(np.random.normal(0, 1, (128, 100)), dtype=torch.float).to(device)
gen_imgs = generator(z)
print(gen_imgs)

save_image(gen_imgs, "images/d.png")

# import matplotlib.pyplot as plt
# gen_imgs = gen_imgs.cpu().detach().numpy()
#
# plt.imshow(gen_imgs[1, :, :,0], cmap='gray')
# plt.axis('off')
# plt.show()
from PIL import Image