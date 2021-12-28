
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.autograd import Variable, grad
import random
import numpy as np
# from torchvision.utils import save_image

from grid_4room_builder import GridWorld

class Env:

    def __init__(self, random_start):

        if random_start:
            start1 = (random.randint(1,9), random.randint(1,9))
            start2 = (random.randint(1,9), random.randint(1,9))
        else:
            start1 = (1,1)
            start2 = (1,1)

        self.grid1 = GridWorld(11, start=start1)
        self.grid2 = GridWorld(11, start=start2)

        self.inp_size = 11*11
        self.num_actions = 4

    def initial_state(self):
        y1 = self.grid1.agent_position
        y2 = self.grid2.agent_position

        x1 = self.grid1.img()
        x2 = self.grid2.img()

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
        self.grid1.step(a1)
        self.grid2.step(a2)

        y1 = self.grid1.agent_position
        y2 = self.grid2.agent_position

        x1 = self.grid1.img()
        x2 = self.grid2.img()


        y1_ = torch.Tensor([y1[0]*11 + y1[1]]).long()
        y2_ = torch.Tensor([y2[0]*11 + y2[1]]).long()

        x1_ = torch.Tensor(x1).float().unsqueeze(0).unsqueeze(0)
        x2_ = torch.Tensor(x2).float().unsqueeze(0).unsqueeze(0)


        return x1_, x2_, y1_, y2_






