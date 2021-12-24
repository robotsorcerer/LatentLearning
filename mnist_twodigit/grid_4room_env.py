
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad
import random
import numpy as np
from torchvision.utils import save_image

from grid_4room_builder import GridWorld

class Env:

    def __init__(self, random_start):

        if random_start:
            start1 = (random.randint(0,10), random.randint(0,10))
            start2 = (random.randint(0,10), random.randint(0,10))
        else:
            start1 = (1,1)
            start2 = (1,1)

        self.grid1 = GridWorld(11, start=start1)
        self.grid2 = GridWorld(11, start=start2)

        self.inp_size = 11*11*3
        self.num_actions = 4

    def initial_state(self):
        #randind1 = random.randint(0,99)
        #randind2 = random.randint(0,99)

        #start_class = 9

        #x1 = torch.cat(self.x_lst[start_class], dim=0).unsqueeze(1)[randind1:randind1+1]
        #y1 = torch.zeros(1).long() + start_class

        #randclass = random.randint(0,9)
        #x2 = torch.cat(self.x_lst[randclass], dim=0).unsqueeze(1)[randind2:randind2+1]
        #y2 = torch.zeros(1).long() + randclass

        #x1 = x1.repeat(1,3,1,1)
        #x2 = x2.repeat(1,3,1,1)

        #c1 = (torch.rand(1,3,1,1).clamp(0.5,1.0)*10.0).round()/10.0
        #c2 = (torch.rand(1,3,1,1).clamp(0.5,1.0)*10.0).round()/10.0

        y1 = self.grid1.agent_position
        y2 = self.grid2.agent_position

        x1 = self.grid1.model
        x2 = self.grid2.model

        c1 = (torch.rand(1,3,1,1).clamp(0.5,1.0)*10.0).round()/10.0
        c2 = (torch.rand(1,3,1,1).clamp(0.5,1.0)*10.0).round()/10.0

        y1 = torch.Tensor([y1[0]*11 + y1[1]]).long()
        y2 = torch.Tensor([y2[0]*11 + y2[1]]).long()
       
        x1 = torch.Tensor(x1).float().unsqueeze(0).unsqueeze(0)
        x2 = torch.Tensor(x2).float().unsqueeze(0).unsqueeze(0)

        return y1,c1,y2,c2,x1,x2

    def random_action(self):
        return torch.randint(0,4,size=(1,))

    def reset(self):
        self.grid1.reset()
        self.grid2.reset()

    def transition(self,a1,a2): 

        #y1 = y1.cuda()
        #y2 = y2.cuda()
        #a1 = a1.cuda()
        #a2 = a2.cuda()

        #y1_ = torch.clamp(y1 + a1,0,9)
        #y2_ = torch.clamp(y2 + a2,0,9)

        self.grid1.step(a1)
        self.grid2.step(a2)

        y1 = self.grid1.agent_position
        y2 = self.grid2.agent_position

        x1 = self.grid1.model
        x2 = self.grid2.model

        y1_ = torch.Tensor([y1[0]*11 + y1[1]]).long()
        y2_ = torch.Tensor([y2[0]*11 + y2[1]]).long()

        x1_ = torch.Tensor(x1).float().unsqueeze(0).unsqueeze(0)
        x2_ = torch.Tensor(x2).float().unsqueeze(0).unsqueeze(0)


        return x1_, x2_, y1_, y2_






