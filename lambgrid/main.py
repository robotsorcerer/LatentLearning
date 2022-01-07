

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
#from encoder import Classifier


from env import Env
from buffer import Buffer
from transition import Transition

import statistics

from value_iter import value_iteration

import argparse


parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, choices=['mnist', 'maze'])
parser.add_argument('--train_iters', type=int, default=2000) #2000

parser.add_argument('--num_rand_initial', type=int, default=500) #2000

parser.add_argument('--random_start', type=str, choices=('true','false'), default='false')

parser.add_argument('--random_policy', type=str, choices=('true', 'false'), default='true')

parser.add_argument('--use_ae', type=str, choices=('true', 'false'), default='false')

parser.add_argument('--log_fh', type=str, default="log.txt")


args = parser.parse_args()

if args.data == 'mnist':
    from env import Env
    from encoders.mlp_pred1 import Classifier as Classifier
elif args.data == 'maze':
    from grid_4room_env import Env
    from encoders.mlp_pred1 import Classifier as Classifier
else:
    raise Exception()


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

        loss += ce(out, a1)
        loss += q_loss

        #print(k_ind, loss, q_loss)

    if False:#print_:
        print('xl_use2', xl_use[0])
        print('xn_use2', xn_use[0])
        print('a', a1)
        print('out', out.shape, out)
        save_image(xl_use[0:10]/10.0, 'xlast2.png')
        save_image(xn_use[0:10]/10.0, 'xnext2.png')
        raise Exception('done')

    ind_last = ind_last.flatten()
    ind_new = ind_new.flatten()

    return out, loss, ind_last, ind_new, a1, y1, y1_, k_offset

ep_length = 5
ep_rand = ep_length

ncodes = 256
genik_maxk = 4

myenv = Env(random_start=(args.random_start=='true'))

def init_model():
    net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=myenv.inp_size*2)

    if torch.cuda.is_available():
        net = net.cuda()

    return net

def init_opt(net):
    opt = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9,0.999))
    return opt

ce = nn.CrossEntropyLoss()

net = init_model()
opt = init_opt(net)

always_random = (args.random_policy == 'true')

num_rand = args.num_rand_initial

mybuffer = Buffer(ep_length=ep_length, max_k=genik_maxk)
transition = Transition(args, ncodes, myenv.num_actions)
transition.reset()


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

        if args.data == 'maze':
            myenv.reset()

        y1,c1,y2,c2,x1,x2 = myenv.initial_state()
        is_initial = False
    else:
        y1 = y1_
        y2 = y2_
        x1 = x1_
        x2 = x2_
    
    if args.data == 'mnist':
        x = torch.cat([x1*c1,x2*c2], dim=3)
    elif args.data == 'maze':
        x = torch.cat([x1,x2], dim=3)
    else:
        raise Exception()

    net.eval()
    #pick actions randomly or with policy
    init_state = net.encode((x*1.0).cuda())
    if always_random or mybuffer.num_ex < num_rand or step >= ep_rand or random.uniform(0,1) < 0.1:
        print('random action')
        a1 = myenv.random_action()
    else:
        print('use policy to pick action!')
        reward = transition.select_goal()
        print('init state abstract', init_state)
        #a1 = transition.select_policy(init_state.cpu().item(), reward)
        a1 = value_iteration(transition.state_transition, ncodes, init_state, reward, max_iter=ep_length)
        a1 = torch.Tensor([a1]).long()
        print('a1', a1)

    a2 = myenv.random_action()

    if args.data == 'mnist':
        x1_, x2_, y1_, y2_ = myenv.transition(a1,a2,y1,y2,c1,c2)
    elif args.data == 'maze':
        x1_, x2_, y1_, y2_ = myenv.transition(a1,a2)
    
    print('example', y1, y1_, a1)

    #make x from x1,x2
    if args.data == 'mnist':
        x_ = torch.cat([x1_*c1,x2_*c2], dim=3)
    elif args.data == 'maze':
        x_ = torch.cat([x1_,x2_], dim=3)
    else:
        raise Exception()

    next_state = net.encode((x_*1.0).cuda())
    transition.update(init_state, next_state, a1, y1, y1_)

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
        num_iter = max(1, args.train_iters//4)
    else:
        num_iter = args.train_iters
    for iteration in range(0,num_iter):

        print_ = iteration==num_iter-1

        do_quantize = iteration >= 500 or mybuffer.num_ex > 150
        
        out, loss, ind_last, ind_new, a1, tr_y1, tr_y1_, _ = update_model(net, mybuffer, print_, do_quantize, reinit_code, 256, None, None)

        if iteration % 100 == 0:
            print('loss', loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        pred = out.argmax(1)
        a1 = a1.cuda()
        pred = pred.cuda()

    net.eval()

    for k in range(0, mybuffer.num_ex, 100):

        ex_lst = list(range(k,min(mybuffer.num_ex, k+100)))

        out, loss, ind_last, ind_new, a1, tr_y1, tr_y1_, _ = update_model(net, mybuffer, print_, True, False, len(ex_lst), ex_lst, klim=1)        

        pred = out.argmax(1)
        a1 = a1.cuda()
        pred = pred.cuda()
        accs.append(torch.eq(pred, a1).float().mean().item())
        transition.update(ind_last, ind_new, a1, tr_y1, tr_y1_)


    accs_all = []
    for k in range(0, mybuffer.num_ex, 100):

        ex_lst = list(range(k,min(mybuffer.num_ex, k+100)))

        out, loss, ind_last, ind_new, a1, tr_y1, tr_y1_, koffset = update_model(net, mybuffer, print_, True, False, len(ex_lst), ex_lst, klim=None)

        pred = out.argmax(1)
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




