import torch
import torch.nn as nn

from quantize import Quantize
#from gumbel_quantize import Quantize
#from quantize_ema import Quantize

from utils import Cutout

from torchvision import transforms as transforms

from torchvision.utils import save_image

import random

class Encoder(nn.Module):

    def __init__(self, ncodes):
        super(Encoder, self).__init__()

        #3*32*64
        self.enc = nn.Sequential(nn.Linear(512,1024), nn.LeakyReLU(), nn.Linear(1024, 512))

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

        do_aug = False
        if self.training and do_aug:
            #print('x shape', x.shape, x.min(), x.max())
            #save_image(x, 'xorig.png')
            x = self.crop(x)
            x = self.resize(x)
            x = self.rotate(x)
            x_aug = self.cutout.apply(x)
            #save_image(x_aug, 'xaug.png')
            #raise Exception('done')
        else:
            x_aug = x

        x_aug = x_aug.reshape((x_aug.shape[0], -1))

        xin = x_aug

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


class Classifier(nn.Module):

    def __init__(self, ncodes):
        super(Classifier, self).__init__()

        self.enc = Encoder(ncodes)

        self.out = nn.Sequential(nn.Linear(512*3, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, 10))
        #self.out = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 3))

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        self.offset_embedding = nn.Embedding(12, 512)

        self.ae_enc = nn.Sequential(nn.Dropout(0.2), nn.Linear(3*32*64,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, 512))
        self.ae_q = Quantize(512,1024,16)
        self.ae_dec = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 3*32*64))

        self.cutout = Cutout(1, 16)

    def ae(self, x):


        print_ = (random.uniform(0,1) < 0.001)

        if print_:
            save_image(x, 'x_in.png')

        x_in = x*1.0

        if self.training:
            x = self.cutout.apply(x)

        x = x.reshape((x.shape[0], -1))
        x_in = x_in.reshape((x_in.shape[0], -1))

        x = self.ae_enc(x).unsqueeze(0)
        z, diff, ind = self.ae_q(x)
        z = z.squeeze(0)
        x = z*1.0
        x = self.ae_dec(x)

        loss = self.mse(x,x_in.detach())*10.0
        loss += diff

        if print_:
            print('rec loss', loss)
            x_rec = x.reshape((x.shape[0], 3, 32, 64))
            save_image(x_rec, 'x_rec.png')

        return loss, z

    def encode(self,x):
        ae_loss_1, z1_low = self.ae(x)
        z1,el_1,ind_1 = self.enc(z1_low, True, False,k=0)
        return ind_1

    #s is of size (bs, 256).  Turn into a of size (bs,3).  
    def forward(self, x, x_next, do_quantize, reinit_codebook=False, k=0, k_offset=None):

        ae_loss_1, z1_low = self.ae(x)
        ae_loss_2, z2_low = self.ae(x_next)

        z1,el_1,ind_1 = self.enc(z1_low, do_quantize, reinit_codebook,k=k)
        z2,el_2,ind_2 = self.enc(z2_low, do_quantize, reinit_codebook,k=k)

        #print('k_offset', k_offset)
        #print('k_offset shape', k_offset.shape)

        #print('z1 shape', z1.shape)

        offset_embed = self.offset_embedding(k_offset)

        #print('offset_embed shape', offset_embed.shape)

        #print('offset embed minmax', offset_embed.min(), offset_embed.max())


        z = torch.cat([z1,z2,offset_embed],dim=1)

        out = self.out(z)

        loss = ae_loss_1 + ae_loss_2 + el_1 + el_2

        return out, loss, ind_1, ind_2, z1, z2









