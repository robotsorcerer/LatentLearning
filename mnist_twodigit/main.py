'''
Two digits: left digit and right digit.  Both digits have random colors which persist over episode.  

Start episode: (x, y1, y2, c1, c2)

Transition: (x,y1,y2,c1,c2,a1,a2) --> (x,y1,y2,c1,c2)

s: (x,y,c).  

s,a,g,r.  

-Train model to predict (s, s') --> a.  

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad
import random
import numpy as np
from torchvision.utils import save_image
from encoder import Classifier

from env import Environment, State

def transition(x,y,y_):
    x_lst = []

    for j in range(0,10):
        x_lst.append(x[y==j])

    x_new = x*0.0

    for j in range(bs):
        x_choices = x_lst[y_[j].item()]
        x_new[j] = x_choices[random.randint(0, x_choices.shape[0]-1)]

    return x_new

net = Classifier().cuda()

ce = nn.CrossEntropyLoss()

opt = torch.optim.Adam(net.parameters(), lr=0.0001)

'''
For each iteration take one step in the environment given the current state.  Keep a replay buffer of all states seen so far.  
Re-encode states to discrete codes.  Record count for each discrete code.  

1.  Take K random steps in environment on each iteration and record state.  Add to buffer.  
2.  

'''

sbuffer = []
steps_per_iter = 100
train_per_iter = 100

myenv = Environment()

curr_state = myenv.init_episode()

for iteration in range(0,1000):

    for step in range(0, steps_per_iter):
        action = random.randint(-1,1)
        next_state,do_reset = myenv.transition(curr_state, action)
    
        sbuffer.append((curr_state, action, next_state))

        if do_reset:
            curr_state = myenv.init_episode()
        else:
            curr_state = next_state


    print('len sbuffer', len(sbuffer))

    for i in range(0, train_per_iter):
        s_train = random.choices(sbuffer,k=256)
        

    raise Exception('done')

for epoch in range(0, 200):

    for (x1,y1),(x2,y2) in zip(train_loader, train_loader):

        x1 = x1.cuda()
        y1 = y1.cuda()
        x2 = x2.cuda()
        y2 = y2.cuda()

        x1 = x1.repeat(1,3,1,1)
        x2 = x2.repeat(1,3,1,1)


        a1 = torch.randint(-1,2,size=(bs,)).cuda()
        a2 = torch.randint(-1,2,size=(bs,)).cuda()

        y1_ = torch.clamp(y1 + a1,0,9)
        y2_ = torch.clamp(y2 + a2,0,9)

        x1_new = transition(x1, y1, y1_)
        x2_new = transition(x2, y2, y2_)
        
        x_last = torch.cat([x1*c1,x2*c2], dim=3)
        x_new = torch.cat([x1_new*c1,x2_new*c2],dim=3)

        out, q_loss, ind_last = net(x_last, x_new, do_quantize = (epoch > 10))

        loss = ce(out, a1+1)
        loss += q_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    print('loss', epoch, loss)

    if ind_last is not None:
        ind_last = ind_last.flatten()
        print(ind_last)
        print(y1)

        for j in range(0,ind_last.max().item() + 1):
            print(j, y1[ind_last==j])

    pred = (out.argmax(1) - 1)
    
    acc = torch.eq(pred, a1).float().mean().item()

    print('acc', acc)

    save_image(x_last, '1.png')
    save_image(x_new, '2.png')
        
        




