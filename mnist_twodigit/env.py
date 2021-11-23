
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad
import random
import numpy as np
from torchvision.utils import save_image

class Env:

    def __init__(self):

        bs = 1

        train_loader = torch.utils.data.DataLoader(datasets.MNIST('data',
                                                         download=True,
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.Resize((32,32)),
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                          ])),
                                           batch_size=bs,
                                            drop_last=True,
                                           shuffle=True)
    

        self.x_lst = []

        for j in range(0,10):
            self.x_lst.append([])

        for (x,y) in train_loader:

            self.x_lst[y[0].item()].append(x[0])

        print(len(self.x_lst))
        for j in range(0, 10):
            self.x_lst[j] = self.x_lst[j][0:100]



