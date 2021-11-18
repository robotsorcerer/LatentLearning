import torch
import torch.nn as nn

from quantize import Quantize

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(3,32,5,2), nn.LeakyReLU(), nn.Conv2d(32,64,5,2),nn.LeakyReLU())
        self.enc2 = nn.Sequential(nn.Linear(4160,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, 256))

        self.q = Quantize(256, 100, 1)

    #x is (bs, 3*32*64).  Turn into z of size (bs, 256).  
    def forward(self, x, do_quantize): 

        x = self.enc1(x)

        x = x.reshape((x.shape[0], -1))

        x = self.enc2(x)

        if do_quantize:
            x = x.unsqueeze(0)
            z_q, diff, ind = self.q(x)
            z_q = z_q.squeeze(0)
        else:
            z_q = x
            diff = 0.0
            ind = None

        return z_q, diff, ind


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.enc = Encoder()

        self.out = nn.Sequential(nn.Linear(256*2, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 10))

    #s is of size (bs, 256).  Turn into a of size (bs,3).  
    def forward(self, x, x_next, do_quantize):

        z1,el_1,ind_1 = self.enc(x, do_quantize)
        z2,el_2,ind_2 = self.enc(x_next, do_quantize)
    
        z = torch.cat([z1,z2],dim=1)

        out = self.out(z)

        return out, el_1 + el_2, ind_1


