import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch



os.makedirs("images/%s" % "edges2shoes", exist_ok=True)
os.makedirs("saved_models/%s" % "edges2shoes", exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_shape = (3, 128, 128)

# Loss functions
mae_loss = torch.nn.L1Loss()

# Initialize generator, encoder and discriminators
generator = Generator(8, input_shape).to(device)
encoder = Encoder(8, input_shape).to(device)
D_VAE = MultiDiscriminator(input_shape).to(device)
D_LR = MultiDiscriminator(input_shape).to(device)


# Initialize weights
generator.apply(weights_init_normal)
D_VAE.apply(weights_init_normal)
D_LR.apply(weights_init_normal)

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=0.0002, betas=(0.5, 0.999))

dataloader = DataLoader(
    ImageDataset("../../data/%s" % "edges2shoes", input_shape),
    batch_size=8,
    shuffle=True,
    num_workers=0,
)
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % "edges2shoes", input_shape, mode="val"),
    batch_size=8,
    shuffle=True,
    num_workers=0,
)


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    generator.eval()
    imgs = next(iter(val_dataloader))
    img_samples = None
    for img_A, img_B in zip(imgs["A"], imgs["B"]):
        # Repeat input image by number of desired columns
        real_A = img_A.view(1, *img_A.shape).repeat(8, 1, 1, 1)
        real_A =torch.FloatTensor(real_A).to(device)

        # Sample latent representations
        sampled_z = torch.FloatTensor(np.random.normal(0, 1, (8, 8))).to(device)
        # Generate samples
        fake_B = generator(real_A, sampled_z)
        # Concatenate samples horisontally
        fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
        img_sample = torch.cat((img_A, fake_B), -1)
        img_sample = img_sample.view(1, *img_sample.shape)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    save_image(img_samples, "images/%s/%s.png" % ("edges2shoes", batches_done), nrow=8, normalize=True)
    generator.train()


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = torch.FloatTensor(np.random.normal(0, 1, (mu.size(0), 8))).to(device)
    z = sampled_z * std + mu
    return z


# ----------
#  Training
# ----------

# Adversarial loss
valid = 1
fake = 0

prev_time = time.time()
for epoch in range(0, 50):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = torch.FloatTensor(batch["A"]).to(device)
        real_B = torch.FloatTensor(batch["B"]).to(device)


        # -------------------------------
        #  Train Generator and Encoder
        # -------------------------------

        optimizer_E.zero_grad()
        optimizer_G.zero_grad()

        # ----------
        # cVAE-GAN
        # ----------

        # Produce output using encoding of B (cVAE-GAN)
        mu, logvar = encoder(real_B)
        encoded_z = reparameterization(mu, logvar)
        fake_B = generator(real_A, encoded_z)

        # Pixelwise loss of translated image by VAE
        loss_pixel = mae_loss(fake_B, real_B)
        # Kullback-Leibler divergence of encoded B
        loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
        # Adversarial loss
        loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

        # ---------
        # cLR-GAN
        # ---------

        # Produce output using sampled z (cLR-GAN)
        sampled_z = torch.FloatTensor(np.random.normal(0, 1, (real_A.size(0), 8))).to(device)
        _fake_B = generator(real_A, sampled_z)
        # cLR Loss: Adversarial loss
        loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)

        # ----------------------------------
        # Total Loss (Generator + Encoder)
        # ----------------------------------

        loss_GE = loss_VAE_GAN + loss_LR_GAN + 10 * loss_pixel + 0.01 * loss_kl

        loss_GE.backward(retain_graph=True)
        optimizer_E.step()

        # ---------------------
        # Generator Only Loss
        # ---------------------

        # Latent L1 loss
        _mu, _ = encoder(_fake_B)
        loss_latent = 0.5 * mae_loss(_mu, sampled_z)

        loss_latent.backward()
        optimizer_G.step()

        # ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------

        optimizer_D_VAE.zero_grad()

        loss_D_VAE = D_VAE.compute_loss(real_B, valid) + D_VAE.compute_loss(fake_B.detach(), fake)

        loss_D_VAE.backward()
        optimizer_D_VAE.step()

        # ---------------------------------
        #  Train Discriminator (cLR-GAN)
        # ---------------------------------

        optimizer_D_LR.zero_grad()

        loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(_fake_B.detach(), fake)

        loss_D_LR.backward()
        optimizer_D_LR.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = 50 * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent: %f] ETA: %s"
            % (
                epoch,
                50,
                i,
                len(dataloader),
                loss_D_VAE.item(),
                loss_D_LR.item(),
                loss_GE.item(),
                loss_pixel.item(),
                loss_kl.item(),
                loss_latent.item(),
                time_left,
            )
        )

        if batches_done % 400== 0:
            sample_images(batches_done)

    if -1 != -1 and epoch % -1 == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % ("edges2shoes", epoch))
        torch.save(encoder.state_dict(), "saved_models/%s/encoder_%d.pth" % ("edges2shoes", epoch))
        torch.save(D_VAE.state_dict(), "saved_models/%s/D_VAE_%d.pth" % ("edges2shoes", epoch))
        torch.save(D_LR.state_dict(), "saved_models/%s/D_LR_%d.pth" % ("edges2shoes", epoch))
