

import torch
import torch.nn as nn

from quantize import Quantize
#from gumbel_quantize import Quantize
#from quantize_ema import Quantize

from utils import Cutout

from torchvision import transforms as transforms

from torchvision.utils import save_image

import random

import numpy as np

class Encoder(nn.Module):

    def __init__(self, args, ncodes, inp_size):
        super(Encoder, self).__init__()

        self.args = args
        #3*32*64
        if self.args.use_ae == 'true':
            self.enc = nn.Sequential(nn.Linear(512,1024), nn.LeakyReLU(), nn.Linear(1024, 512))
        else:
            self.enc = nn.Sequential(nn.Linear(inp_size,1024), nn.LeakyReLU(), nn.Linear(1024, 512))

        self.qlst = []

        self.cutout = Cutout(1, 16)

        for nf in [1,8,32]:
            self.qlst.append(Quantize(512, ncodes, nf))

        self.qlst = nn.ModuleList(self.qlst)

        self.crop = transforms.RandomCrop((30,60))
        self.resize = transforms.Resize((32,64))
        self.rotate = transforms.RandomRotation((-5,5))
        self.color = transforms.ColorJitter(0.1,0.1,0.1,0.1)


    #x is (bs, 3*32*64).  Turn into z of size (bs, 256).
    def forward(self, x, do_quantize, reinit_codebook=False, k=0):

        xin = x.reshape((x.shape[0], -1))


        x = self.enc(xin)

        if do_quantize:
            x = x.unsqueeze(0)
            q = self.qlst[k]
            z_q, diff, ind = q(x, reinit_codebook)
            z_q = z_q.squeeze(0)
        else:
            z_q = x
            diff = 0.0
            ind = None

        return z_q, diff, ind


