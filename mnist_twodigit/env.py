'''

State Class.  Store buffer as a list of (s, a, s') 3-tuples.  
  -State2Image.  
  -Store color_left, color_right, class_left, class_right, image_left, image_right.  


Environment Class.  Can produce transition (s,a) --> s'.  Can initialize s.  
    -Stores all of the examples.  
    -
'''

import numpy
import torch

from torchvision import datasets, transforms


import random

class State:

    def __init__(self,c1,c2,y1,y2,x1,x2,t):
        self.c1 = c1
        self.c2 = c2
        self.y1 = y2
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2
        self.t = t

    def to_image(self):
        return torch.cat([self.x1*self.c1, self.x2*self.c2],dim=3)

class Environment:
    def __init__(self):

        train_loader = torch.utils.data.DataLoader(datasets.MNIST('data',
                                                         download=True,
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.Resize((32,32)),
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                          ])),batch_size=256,drop_last=True,shuffle=True)

        
        self.all_x = []
        self.all_y = []

        for x,y in train_loader:
            self.all_x.append(x)
            self.all_y.append(y)

        self.all_x = torch.cat(self.all_x,dim=0)
        self.all_y = torch.cat(self.all_y,dim=0)

        self.x_lst = []

        num_classes = 10
        for j in range(num_classes):
            self.x_lst.append(self.all_x[self.all_y==j])

        self.max_len = 15

    #for j in range(bs):
    #    x_choices = x_lst[y_[j].item()]
    #    x_new[j] = x_choices[random.randint(0, x_choices.shape[0]-1)]

    #return x_new

    '''
        class random (8,9) on each side.  color random on each side.  
    '''
    def init_episode(self):
        y1 = random.randint(8,9)
        y2 = random.randint(8,9)

        c1 = torch.rand(1,3,1,1)
        c2 = torch.rand(1,3,1,1)

        x1 = self.get_image_y(y1)

        x2 = self.get_image_y(y2)

        t = 0
        return State(c1,c2,y1,y2,x1,x2,t)

    def get_image_y(self, y):
        x = self.x_lst[y]
        x = x[random.randint(0, len(x)-1)]

        return x

    '''
        change y1 based on a1
    '''
    def transition(self,state,a1):

        a2 = random.randint(-1,1)

        y1 = numpy.clip(state.y1 + a1, 0, 9)
        y2 = numpy.clip(state.y2 + a2, 0, 9)

        x1 = self.get_image_y(y1)
        x2 = self.get_image_y(y2)

        new_state = State(state.c1, state.c2, y1, y2, x1, x2, state.t+1)

        if state.t > self.max_len:
            end_ep = True
        else:
            end_ep = False

        return new_state, end_ep






