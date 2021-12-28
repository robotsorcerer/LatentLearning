import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
import random
import numpy as np
from encoder import Classifier

from buffer import Buffer
from transition import Transition

import statistics

from value_iteration import value_iteration

import argparse
from grid_4room_env import Env

from visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld, MazeWorld, SpiralWorld, LoopWorld
from visgrid.utils import get_parser, MI
from visgrid.sensors import *
from visgrid.gridworld.distance_oracle import DistanceOracle


from gridworld.gridworld_wrapper import GridWorldWrapper


parser = argparse.ArgumentParser(description='Maze Task', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', type=str, choices=['maze', 'vis-maze', 'minigrid'])

parser.add_argument('--train_iters', type=int, default=5000) #2000

parser.add_argument('--num_rand_initial', type=int, default=2000) #2000

parser.add_argument('--random_start', type=str, choices=('true','false'), default='false')

parser.add_argument('--random_policy', type=str, choices=('true', 'false'), default='true')

parser.add_argument('--use_ae', type=str, choices=('true', 'false'), default='false')


# arguments for visgrid mazes
parser.add_argument('--walls', type=str, default='empty', choices=['empty', 'maze', 'spiral', 'loop'],
                    help='The wall configuration mode of gridworld')

parser.add_argument('-r','--rows', type=int, default=11, help='Number of gridworld rows')

parser.add_argument('-c','--cols', type=int, default=11, help='Number of gridworld columns')

parser.add_argument("--noise_type", type=str, default=None, choices=[None, 'ising', 'ellipse', 'tv'], help='Exo noise to observations')

parser.add_argument("--noise_stationarity", type=str, default='stationary', choices=['non-stationary', 'stationary'], help='resample noise every step?')
# Noise-specific arguments
parser.add_argument('--ising_beta', type=float, default=0.5,
                    help='Ising model\'s beta parameter')

parser.add_argument('--obs_type', type=str, default='pixels', choices=['pixels', 'high_dim'],
                    help='Which type of observation to use')

args = parser.parse_args()



#% ------------------ Define MDP from VisGrid Mazes ------------------
if args.walls == 'maze':
    env = MazeWorld.load_maze(rows=args.rows, cols=args.cols, seed=args.seed)
    env_name = 'mazeworld'
elif args.walls == 'spiral':
    env = SpiralWorld(rows=args.rows, cols=args.cols)
    env_name = 'spiralworld'
elif args.walls == 'loop':
    env = LoopWorld(rows=args.rows, cols=args.cols)
    env_name = 'loopworld'
else:
    env = GridWorld(rows=args.rows, cols=args.cols, noise_type=args.noise_type, ising_beta=args.ising_beta)
    env_name = 'gridworld'


if args.data == 'maze':
    myenv = Env(random_start=(args.random_start=='true'))
elif args.data == 'vis-maze':
    myenv = env
elif args.data == 'minigrid':
    myenv = GridWorldWrapper.make_env("twogrids")
    env = myenv
    myenv.num_actions = len(env.actions)

    # action = random.randint(0, 4)
    # obs, reward, done, info = env.step(action)
else:
    raise Exception()

rows = args.rows
cols = args.cols

ep_length = 5
ep_rand = ep_length

ncodes = 256
genik_maxk = 4


def update_model(model, mybuffer, print_, do_quantize, reinit_codebook,bs,batch_ind=None, klim=None): 

    a1, y1, y1_, x_last, x_new, k_offset = mybuffer.sample_batch(bs, batch_ind, klim=klim)
    loss = 0.0
    for k_ind in [2,1,0]:
        xl_use = x_last*1.0
        xn_use = x_new*1.0

        out, q_loss, ind_last, ind_new, z1, z2 = model(xl_use, xn_use, do_quantize = True, reinit_codebook = reinit_codebook, k=k_ind, k_offset=k_offset)

        loss += ce(out, a1)
        loss += q_loss

    if False:#print_:
        print('xl_use2', xl_use[0])
        print('xn_use2', xn_use[0])
        print('a', a1)
        print('out', out.shape, out)
        raise Exception('done')

    ind_last = ind_last.flatten()
    ind_new = ind_new.flatten()

    return out, loss, ind_last, ind_new, a1, y1, y1_, k_offset



def init_model():
    if args.data=='maze':
        net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=myenv.inp_size*2)
    elif args.data == 'vis-maze':
        if args.obs_type == 'pixels':
            net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=80 * 160 * 3)
        elif args.obs_type == 'high_dim':
            net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=11 * 11 * 2)
    elif args.data == 'minigrid':
        net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=60 * 120 * 3)

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
transition = Transition(ncodes, myenv.num_actions)
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

        # if args.data == 'maze':
        myenv.reset()

        if args.data == 'maze':
            y1,c1,y2,c2,x1,x2 = myenv.initial_state()
            
        elif args.data == 'vis-maze': ##### TODO HERE : CONSTRUCT THE TWO-MAZE VIS-GRID
            init_state = env.get_state()
            init_state2 = env.get_state()

            if args.obs_type == "pixels":
                x1, x1_noisy = env.get_obs(change_noise=args.noise_stationarity=='non-stationary')
            else:
                x1 = env.img() #### TODO HERE : Add flexibility for exo noise in observations too

            y1 = init_state
            y1 = torch.Tensor([y1[0]*11 + y1[1]]).long()

            y2 = init_state2
            y2 = torch.Tensor([y2[0]*11 + y2[1]]).long()

            x1 = env.get_observation(x1, args.obs_type)

        elif args.data == 'minigrid':
            img, info1 = env.reset()
            img2, info2 = env.reset()


            true_state1 = info1['state']
            true_state2 = info2['state']

            y1 = np.array([ true_state1[0], true_state1[1]  ])
            y2 = np.array([ true_state2[0], true_state2[1]  ])

            y1 = torch.Tensor([y1[0]*11 + y1[1]]).long()
            y2 = torch.Tensor([y2[0]*11 + y2[1]]).long()

        is_initial = False


    if args.data == 'maze':
        x = torch.cat([x1,x2], dim=3)
    elif args.data == 'vis-maze':
        if args.obs_type == 'high_dim':
            x = torch.cat([x1,x1], dim=3)
        else :
            x = torch.cat([x1,x1], dim=2)
    elif args.data == 'minigrid':
        # x = torch.tensor(img)
        x = torch.Tensor(img).float().unsqueeze(0)
    else:
        raise Exception()
    net.eval()

    #pick actions randomly or with policy
    init_state = net.encode((x*1.0))

    if always_random or mybuffer.num_ex < num_rand or step >= ep_rand or random.uniform(0,1) < 0.1:
        print('random action')
        a1 = myenv.random_action()
    else:
        print('use policy to pick action!')
        reward = transition.select_goal()

        print('init state abstract', init_state)

        ## a1 = transition.select_policy(init_state.cpu().item(), reward)
        a1 = value_iteration(transition.state_transition, ncodes, init_state, reward, max_iter=ep_length)

    a1 = torch.Tensor([a1]).long()
    print('a1', a1)

    a2 = myenv.random_action()


    if args.data == 'maze':
        x1_, x2_, y1_, y2_ = myenv.transition(a1,a2)

    elif args.data == 'vis-maze':
        x1_, reward1, _ = env.step(a1.item())
        x2_, reward2, _ = env.step(a2.item())

        y1_ = x1_
        y2_ = x2_

        y1_ = torch.Tensor([y1_[0]*11 + y1_[1]]).long()
        y2_ = torch.Tensor([y2_[0]*11 + y2_[1]]).long()
        # y1_, y2_ = env.states_pos_to_int(y1_, y2_, args.rows, args.cols) 

        if args.obs_type == "pixels":
            x1_, x1_noisy_ = env.get_obs(change_noise=args.noise_stationarity=='non-stationary')
        else:
            x1_ = env.img() #### TODO HERE : Add flexibility for exo noise in observations too

        x1 = env.get_observation(x1_, args.obs_type)

    elif args.data == 'minigrid':
        obs1, reward1, done, info1 = env.step(a1)
        obs2, reward2, done, info2 = env.step(a2)

        true_state1 = info1['state']
        true_state2 = info2['state']

        y1_ = np.array([ true_state1[0], true_state1[1]  ])
        y2_ = np.array([ true_state2[0], true_state2[1]  ])

        y1_ = torch.Tensor([y1_[0]*11 + y1_[1]]).long()
        y2_ = torch.Tensor([y2_[0]*11 + y2_[1]]).long()

    else:
        raise Exception()    

    # print('example', y1, y1_, a1)

    #make x from x1,x2
    if args.data == 'maze':
        x_ = torch.cat([x1_,x2_], dim=3)
    elif args.data == 'vis-maze':
        if args.obs_type == 'high_dim':
            x_ = torch.cat([x1,x1], dim=3)
        else :
            x_ = torch.cat([x1,x1], dim=2)
    elif args.data == 'minigrid':
        x_ = torch.Tensor(obs1).float().unsqueeze(0)
    else:
        raise Exception()

    next_state = net.encode((x_*1.0))


    transition.update(init_state, next_state, a1, y1, y1_)

    # mybuffer.add_example(a1, y1, y1_, c1, y2, y2_, c2, x, x_, step)
    mybuffer.add_example(a1, y1, y1_, y2, y2_, x, x_, step)

    print('my buffer numex', mybuffer.num_ex)

    if mybuffer.num_ex < num_rand or mybuffer.num_ex % 100 != 0:
        continue

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
        # a1 = a1.cuda()
        # pred = pred.cuda()

    net.eval()

    for k in range(0, mybuffer.num_ex, 100):

        ex_lst = list(range(k,min(mybuffer.num_ex, k+100)))

        out, loss, ind_last, ind_new, a1, tr_y1, tr_y1_, _ = update_model(net, mybuffer, print_, True, False, len(ex_lst), ex_lst, klim=1)        

        pred = out.argmax(1)
        # a1 = a1.cuda()
        # pred = pred.cuda()
        accs.append(torch.eq(pred, a1).float().mean().item())
        transition.update(ind_last, ind_new, a1, tr_y1, tr_y1_)


    accs_all = []

    for k in range(0, mybuffer.num_ex, 100):

        ex_lst = list(range(k,min(mybuffer.num_ex, k+100)))

        out, loss, ind_last, ind_new, a1, tr_y1, tr_y1_, koffset = update_model(net, mybuffer, print_, True, False, len(ex_lst), ex_lst, klim=None)

        pred = out.argmax(1)
        # a1 = a1.cuda()
        # pred = pred.cuda()
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




