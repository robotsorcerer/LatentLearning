import torch
import torch.nn as nn

from quantize import Quantize
#from gumbel_quantize import Quantize
#from quantize_ema import Quantize

from utils import Cutout

from torchvision import transforms as transforms

from torchvision.utils import save_image

class Encoder(nn.Module):

    def __init__(self, ncodes):
        super(Encoder, self).__init__()

        self.enc = nn.Sequential(nn.Dropout(0.2), nn.Linear(3*32*64,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, 512))

        self.qlst = []
        self.kemb = []

        self.cutout = Cutout(1, 16)

        for nf in [1,16,64,256]:
            #self.qlst.append(Quantize(512, ncodes, nf))
            self.qlst.append(Quantize(512,ncodes))
            self.kemb.append(nn.Embedding(1, 3*32*64))

        self.qlst = nn.ModuleList(self.qlst)
        self.kemb = nn.ModuleList(self.kemb)
        
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

        emb = self.kemb[k](torch.zeros(x_aug.shape[0],1).long().cuda())

        xin = x_aug + emb.squeeze(1)

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

        self.proj_layer = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 1024), nn.LeakyReLU(), nn.Linear(1024,3*32*64))
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def proj_loss(self, z1, x):

        x = x.reshape((x.shape[0], -1))

        pred = self.proj_layer(torch.cat([z1],dim=1))

        loss = self.mse(pred,x)*1.0

        return loss

    #s is of size (bs, 256).  Turn into a of size (bs,3).  
    def forward(self, x, x_next, do_quantize, reinit_codebook=False, k=0):

        z1,el_1,ind_1 = self.enc(x, do_quantize, reinit_codebook,k=k)
        z2,el_2,ind_2 = self.enc(x_next, do_quantize, reinit_codebook,k=k)

        z = torch.cat([z1,z2,z1-z2],dim=1)

        out = self.out(z)

        return out, el_1 + el_2, ind_1, ind_2, z1, z2









