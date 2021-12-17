

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
from buffer import Buffer
from transition import Transition

import statistics

from value_iteration import value_iteration

bs = 100

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


def update_model(model, mybuffer, print_, do_quantize, reinit_codebook,bs,batch_ind=None, klim=None): 

    a1, y1, y1_, x_last, x_new, k_offset = mybuffer.sample_batch(bs, batch_ind, klim=klim)


    if torch.cuda.is_available():
        x_last.cuda()
        x_new.cuda()
        a1.cuda()
        y1.cuda()
        y1_.cuda()
        model.cuda()

    loss = 0.0
    for k_ind in [2,1,0]:
        xl_use = x_last*1.0
        xn_use = x_new*1.0



        out, q_loss, ind_last, ind_new, z1, z2 = model(xl_use, xn_use, do_quantize = True, reinit_codebook = reinit_codebook, k=k_ind, k_offset=k_offset)

        #if k_ind == 0:
        #    pl = model.proj_loss(z1, xl_use)
        #    pl2 = model.proj_loss(z2, xn_use)
        #    loss += pl
        #    loss += pl2

        loss += ce(out, a1+1)
        loss += q_loss

        #print(k_ind, loss, q_loss)

    #if print_:
    #    print('a', a1)
    #    print('out', out.shape, out)
        #save_image(xl_use, 'xlast.png')
        #save_image(xn_use, 'xnext.png')


    ind_last = ind_last.flatten()
    ind_new = ind_new.flatten()

    return out, loss, ind_last, ind_new, a1, y1, y1_, k_offset

ncodes = 128
genik_maxk = 19

def init_model():
    net = Classifier(ncodes=ncodes, maxk=genik_maxk)

    if torch.cuda.is_available():
        net = net.cuda()

    return net

def init_opt(net):
    opt = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9,0.999))
    return opt

ce = nn.CrossEntropyLoss()

net = init_model()
opt = init_opt(net)

always_random = False

num_rand = 100
ep_length = 20
ep_rand = ep_length

myenv = Env()
mybuffer = Buffer(ep_length=ep_length, max_k=genik_maxk)
transition = Transition(ncodes)

is_initial = True
step = 0

reinit_code = False

for env_iteration in range(0, 200000):

    #do one step in env.  

    if step == ep_length:
        print('reinit env')
        is_initial=True
        step = 0
        ep_rand = random.randint(ep_length//2, ep_length) #after this step in episode follow random policy
    else:
        step += 1

    if is_initial:
        y1,c1,y2,c2,x1,x2 = myenv.initial_state()
        is_initial = False
    else:
        y1 = y1_
        y2 = y2_
        x1 = x1_
        x2 = x2_
    
    x = torch.cat([x1*c1,1*x2*c2], dim=3)

    net.eval()
    #pick actions randomly or with policy
    if always_random or mybuffer.num_ex < num_rand or step >= ep_rand or random.uniform(0,1) < 0.2:
        print('random action')
        a1 = torch.randint(-1,2,size=(1,))
    else:
        print('use policy to pick action!')
        reward = transition.select_goal()
        init_state = net.encode((x*1.0).cuda())
        print('init state abstract', init_state)
        #a1 = transition.select_policy(init_state.cpu().item(), reward)
        a1 = value_iteration(transition.state_transition, ncodes, init_state, reward, max_iter=ep_length)
        a1 = torch.Tensor([a1]).long()
        print('a1', a1)

    a2 = torch.randint(-1,2,size=(1,))

    x1_, x2_, y1_, y2_ = myenv.transition(a1,a2,y1,y2,c1,c2)

    print('example', y1, y1_, a1)

    #make x from x1,x2
    x_ = torch.cat([x1_*c1,1*x2_*c2], dim=3)

    mybuffer.add_example(a1, y1, y1_, c1, y2, y2_, c2, x, x_, step)

    print('my buffer numex', mybuffer.num_ex)

    if mybuffer.num_ex < num_rand or mybuffer.num_ex % 100 != 0:
        continue

    #net = init_model()
    opt = init_opt(net)
    transition.reset()

    net.train()
    accs = []
    if mybuffer.num_ex > 150:
        num_iter = 500
    else:
        num_iter = 2000
    for iteration in range(0,num_iter):

        print_ = iteration==num_iter-1

        do_quantize = iteration >= 500 or mybuffer.num_ex > 150
        
        out, loss, ind_last, ind_new, a1, tr_y1, tr_y1_, _ = update_model(net, mybuffer, print_, do_quantize, reinit_code, 128, None, None)

        opt.zero_grad()
        loss.backward()
        opt.step()

        pred = (out.argmax(1) - 1)
        a1 = a1.cuda()
        pred = pred.cuda()

    net.eval()

    for k in range(0, mybuffer.num_ex, 100):

        ex_lst = list(range(k,min(mybuffer.num_ex, k+100)))

        out, loss, ind_last, ind_new, a1, tr_y1, tr_y1_, _ = update_model(net, mybuffer, print_, True, False, len(ex_lst), ex_lst, klim=1)        

        pred = (out.argmax(1) - 1)
        a1 = a1.cuda()
        pred = pred.cuda()
        accs.append(torch.eq(pred, a1).float().mean().item())
        transition.update(ind_last, ind_new, a1, tr_y1, tr_y1_)


    accs_all = []
    for k in range(0, mybuffer.num_ex, 100):

        ex_lst = list(range(k,min(mybuffer.num_ex, k+100)))

        out, loss, ind_last, ind_new, a1, tr_y1, tr_y1_, koffset = update_model(net, mybuffer, print_, True, False, len(ex_lst), ex_lst, klim=None)

        pred = (out.argmax(1) - 1)
        a1 = a1.cuda()
        pred = pred.cuda()
        accs_all.append(torch.eq(pred, a1).float().mean().item())


        if False:
            print('genik')
            for j in range(0,koffset.shape[0]):
                print('offsets', koffset[j])
                print('action', (a1)[j])
                print('last_y', tr_y1[j])
                print('new_y', tr_y1_[j])
                print('last_ind', ind_last[j])
                print('new_ind', ind_new[j])
                print('------------------------------')
 

    transition.print_codes()
    transition.print_modes()
    print('loss', env_iteration, loss)
    print('acc-1', sum(accs)/len(accs))
    print('acc-k', sum(accs_all)/len(accs_all))
    
    #raise Exception('done')    




