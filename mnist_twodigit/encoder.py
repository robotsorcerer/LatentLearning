import torch
import torch.nn as nn

from quantize import Quantize
#from gumbel_quantize import Quantize

class Encoder(nn.Module):

    def __init__(self, ncodes):
        super(Encoder, self).__init__()

        self.enc = nn.Sequential(nn.Linear(3*32*64,512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512, 512))

        self.q = Quantize(512, ncodes)

    #x is (bs, 3*32*64).  Turn into z of size (bs, 256).  
    def forward(self, x, do_quantize): 

        x = x.reshape((x.shape[0], -1))

        x = self.enc(x)

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

    def __init__(self, ncodes):
        super(Classifier, self).__init__()

        self.enc = Encoder(ncodes)

        self.out = nn.Sequential(nn.Linear(512*2, 512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512, 3))
        #self.out = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 3))

    #s is of size (bs, 256).  Turn into a of size (bs,3).  
    def forward(self, x, x_next, do_quantize):

        z1,el_1,ind_1 = self.enc(x, do_quantize)
        z2,el_2,ind_2 = self.enc(x_next, do_quantize)

        z = torch.cat([z1,z2],dim=1)

        out = self.out(z)

        return out, el_1 + el_2, ind_1, ind_2




