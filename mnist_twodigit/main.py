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
import copy
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

opt = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5,0.999), weight_decay=1e-4)

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

for iteration in range(0,100):

    print('iteration', iteration)

    for step in range(0, steps_per_iter):
        action = random.randint(-1,1)
        next_state,do_reset = myenv.transition(curr_state, action)
    
        sbuffer.append((curr_state, action, next_state))


        if do_reset:
            curr_state = myenv.init_episode()
        else:
            curr_state = copy.deepcopy(next_state)

    if len(sbuffer) > 40000:
        sbuffer = sbuffer[-10000:]

    acc_lst = []

    for i in range(0, train_per_iter):
        s_train = random.choices(sbuffer,k=128)

        x_lst = []
        xn_lst = []       
        a_lst = []
        y1_lst = []

        for j in range(0, len(s_train)):
            s,a,sn = s_train[j]

            y1_lst.append(torch.Tensor([s.y1]).long())
            x_lst.append(torch.Tensor(s.to_image()))
            a_lst.append(torch.Tensor([a]).long())
            xn_lst.append(torch.Tensor(sn.to_image()))

        x = torch.cat(x_lst,dim=0).cuda()
        a1 = torch.cat(a_lst,dim=0).cuda()
        xn = torch.cat(xn_lst,dim=0).cuda()
        y1 = torch.cat(y1_lst,dim=0).cuda()


        out, q_loss, ind_last = net(x, xn, do_quantize = (iteration > 5))

        #loss = ce(out, a1+1)
        loss = ce(out, y1)
        loss += q_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        #pred = (out.argmax(1) - 1)
        pred = out.argmax(1)

        acc = torch.eq(pred, y1).float().mean().item()

        acc_lst.append(acc)

    if iteration % 5 == 0:
        print(iteration, 'len sbuffer', len(sbuffer))
        print('loss', loss)

        if ind_last is not None:
            ind_last = ind_last.flatten()
            print(ind_last)
            print(y1)

            print(ind_last.shape, y1.shape)

            for j in range(0,ind_last.max().item() + 1):
                print(j, y1[ind_last==j])


        print('acc', sum(acc_lst)/len(acc_lst))

        print(y1)
        print(a1)
        print('pred', pred)
        #save_image(x, '1.png')
        #save_image(xn, '2.png')
        #raise Exception('done')
        




