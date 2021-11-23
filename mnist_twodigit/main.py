

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

from env import Env

bs = 256

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



#def colorize(x): 
#    x = x.repeat(1,3,1,1)
#    bs = x.shape[0]
 
#    c = torch.rand(bs,3,1,1)

#    x *= c

#    return x, c

#def transition(x,y1,y2,c1,c2,a1,a2):

#Given (cat(x1,x2), cat(x1,x2)_ --> classify a.  ).  

def transition(x,y,y_,myenv):
    x_new = x*0.0

    for j in range(bs):
        x_choices = myenv.x_lst[y_[j].item()]
        
        x_new[j] = x_choices[random.randint(0, len(x_choices)-1)]

    return x_new



net = Classifier()

if torch.cuda.is_available():
    net = net.cuda()

ce = nn.CrossEntropyLoss()

opt = torch.optim.Adam(net.parameters(), lr=0.0001)

myenv = Env()


for epoch in range(0, 200):

    accs = []
    state_transition = torch.zeros(30,3,30)

    tr_lst = []
    for j in range(0,30):
        tr_lst.append([])

    for (x1,y1),(x2,y2) in zip(train_loader, train_loader):

        x1 = x1
        y1 = y1
        x2 = x2
        y2 = y2

        if torch.cuda.is_available():
            x1 = x1.cuda()
            y1 = y1.cuda()
            x2 = x2.cuda()
            y2 = y2.cuda()

        x1 = x1.repeat(1,3,1,1)
        x2 = x2.repeat(1,3,1,1)

        c1 = torch.rand(bs,3,1,1)
        c2 = torch.rand(bs,3,1,1)

        a1 = torch.randint(-1,2,size=(bs,))
        a2 = torch.randint(-1,2,size=(bs,))

        if torch.cuda.is_available():
            c1 = c1.cuda()
            c2 = c2.cuda()
            a1 = a1.cuda()
            a2 = a2.cuda()

        y1_ = torch.clamp(y1 + a1,0,9)
        y2_ = torch.clamp(y2 + a2,0,9)

        x1_new = transition(x1, y1, y1_,myenv)
        x2_new = transition(x2, y2, y2_,myenv)        

        x_last = torch.cat([x1*c1,x2*c2], dim=3)
        x_new = torch.cat([x1_new*c1,x2_new*c2],dim=3)

        out, q_loss, ind_last, ind_new = net(x_last, x_new, do_quantize = (epoch > 10))

        loss = ce(out, a1+1)
        loss += q_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        pred = (out.argmax(1) - 1)
        accs.append(torch.eq(pred, a1).float().mean().item())

        if ind_last is not None:
            for j in range(0, ind_last.shape[0]):
                state_transition[ind_last[j], a1[j], ind_new[j]] += 1

            for j in range(0,ind_last.max().item() + 1):
                tr_lst[j] += y1[ind_last.flatten()==j].data.cpu().numpy().tolist()

    print('loss', epoch, loss)

    if ind_last is not None:
        ind_last = ind_last.flatten()
        ind_new = ind_new.flatten()

        if epoch % 5 == 0:
            print(ind_last)
            print(y1)

            mode_lst = []
            for j in range(0,ind_last.max().item() + 1):
                #print(j, y1[ind_last==j])
                random.shuffle(tr_lst[j])
                mode_lst.append(torch.Tensor(tr_lst[j]).argmax())
                print(j, tr_lst[j][0:30])

            print('state transition matrix!')
            for a in range(0,3):
                for k in range(0,state_transition.shape[0]):
                    if state_transition[k,a].sum().item() > 0:
                        print(mode_lst[k], a-1, 'argmax', mode_lst[state_transition[k,a].argmax()], 'num', state_transition[k,a].sum().item())
    

    print('acc', sum(accs)/len(accs))

    #save_image(x_last, '1.png')
    #save_image(x_new, '2.png')
        
        
