from noisy_state_abstractions.algorithms.dqn_infomax.utils import set_seed
from noisy_state_abstractions.algorithms.dqn_infomax.loggers import Logger, SummaryWriter_X
from noisy_state_abstractions.algorithms.dqn_infomax.buffer import ReplayMemoryOptimized
from noisy_state_abstractions.envs.wrappers import HistoryWrapper, WarpFramePocMan, IsingWrapper, PixelRenderer

from noisy_state_abstractions.envs.four_rooms import FourRooms

from noisy_state_abstractions.algorithms.dqn_infomax.infomax_agent import InfoMaxAgent
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import namedtuple, defaultdict
from copy import deepcopy
from collections import deque
import cv2
import os

import uuid

import gym
import gym_minigrid
from gym_minigrid.wrappers import  FullyObsWrapper,RGBImgObsWrapper

import argparse


def soft_update(model, target, tau):
    for target_param, local_param in zip(target.parameters(), model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def eval_policy(n_eval, env, model, T_max, args):
    reward_list = []
    for i_tran in range(n_eval):
        state = (torch.FloatTensor(env.reset()).to(model.device) / 255.)[:, :,
                :args['frame_stack'] * args['num_channels']]
        traj_reward = 0
        for t in range(T_max):
            action = model.select_greedy_action(state)

            next_state, reward, done, _ = env.step(action)
            traj_reward += reward
            next_state = (torch.FloatTensor(next_state).to(model.device) / 255.)[:, :,
                         :args['frame_stack'] * args['num_channels']]

            # Move to the next state
            state = next_state
            if done:
                break
        reward_list.append(traj_reward)
    return np.mean(reward_list)
    

def main(cmd_args):
    seed = int(np.random.randint(0,100000000))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    args = {
            'env_name':cmd_args['env_name'],
            'device':'cuda:0' if torch.cuda.is_available() else 'cpu',
            'seed':seed,

            # Evaluation
            'weight_save_interval':100,
            'test_interval':100,
            'test_episodes':10,
            'N_traj':22000,
            'warmup_steps':1000,
            'T_max':500,
            'replay_size':1000000,
            'H':5,

            # DQN
            'BATCH_SIZE': 128,
            'GAMMA':0.95,
            'EPS_START':0.1,
            'EPS_END':0.01,
            'EPS_DECAY':10000,
            'target_update_freq': 100,
            'tau': 1.0,

            # InfoMax
            'lambda_LL':cmd_args['lambda_LL'],
            'lambda_GL':cmd_args['lambda_GL'],
            'lambda_LG':cmd_args['lambda_LG'],
            'lambda_GG':cmd_args['lambda_GG'],
            'lambda_rl':cmd_args['lambda_rl'], 
            'data_aug':0,
            'ema_moco':0,
            'predictive_k': 2
            }

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    print(cmd_args)
    log_dir = 'runs/LL_{}_target100_Loss_'.format(cmd_args['lambda_LL']) + current_time
    # log_dir = 'runs/test_' + current_time
    logger = SummaryWriter(log_dir=log_dir)

    weight_save_window = 50
    if args['env_name'] == 'PocManFullRGB-v0':
        env_args = {
                    'id':cmd_args['env_name'],
                    'harmless_ghosts' : [], # [i for i in range(4) if i!=j] for j in range(2)
                    'wall_place_prob': 0.0,
                    'food_place_prob':0.2,
                    'ghost_random_move_prob':[0.,0.,0.,0.]
                    }

        args = {**args,**env_args}

        args['ising_beta'] = 4.
        args['ising'] = True
        args['frame_stack'] = 4
        args['num_channels'] = 3

        env = HistoryWrapper( IsingWrapper(
                                            WarpFramePocMan( gym.make(**env_args) ,
                                                            21*2,19*2,grayscale=(args['num_channels'] == 1)
                                                            ), 21*2,19*2, beta=args['ising_beta']),
                                                            X=args['frame_stack']*args['H'])
        env_noiseless = HistoryWrapper( WarpFramePocMan( gym.make(**env_args) ,
                                                            21*2,19*2,grayscale=(args['num_channels'] == 1)),
                                                            X=args['frame_stack']*args['H'])
    elif cmd_args['env_name'] == 'FourRooms':
        args['frame_stack'] = 1
        args['num_channels'] = 1
        env = WarpFramePocMan(FourRooms(goal=(10,10),viz_params=['pixel']),width=21*2,height=19*2,grayscale=True)
    else:
        args['frame_stack'] = 1
        args['num_channels'] = 1
        env = WarpFramePocMan(PixelRenderer(gym.make(cmd_args['env_name'])),width=21*2,height=19*2,grayscale=True)

    exp_name = 'DQN'+ ('_vanilla' if args['lambda_LL']==args['lambda_LG']==args['lambda_GL']==args['lambda_GG']==0 else '_nce')

    exp_path = os.path.join(cmd_args['env_name'], exp_name+str(seed) )

    """
    To run:
    =======Vanilla DQN=======

    NVIDIA_VIDIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python train_pacman.py 

    NVIDIA_VIDIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python trian_pacman.py --env_name MountainCar-v0

    =======InfoMax=======

    NVIDIA_VIDIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python train_pacman.py --lambda_LL=1

    NVIDIA_VIDIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python trian_pacman.py --env_name MountainCar-v0 --lambda_LL=1


    
    """

    N_traj = args['N_traj']
    T_max = args['T_max']
    EPS_START = args['EPS_START']
    EPS_END = args['EPS_END']
    EPS_DECAY = args['EPS_DECAY']
    weight_save_interval = args['weight_save_interval'] # save every X episodes

    model = InfoMaxAgent(np.zeros(shape=(env.observation_space.shape[0],env.observation_space.shape[1],args['frame_stack']*args['num_channels'])),
                            env.action_space,device,EPS_START,EPS_END,EPS_DECAY, tau=args['tau'],
                            batch_size = args['BATCH_SIZE'],
                            num_channels=args['num_channels'],
                            frame_stack=args['frame_stack'],
                            gamma=args['GAMMA'],
                            lambda_LL=args['lambda_LL'],
                            lambda_LG=args['lambda_LG'],
                            lambda_GL=args['lambda_GL'],
                            lambda_GG=args['lambda_GG'],
                            lambda_rl=args['lambda_rl'],
                            predictive_k=args['predictive_k'],
                            encoder='infoNCE_Mnih_84x84_action_FILM')
    model = model.to(model.device)

    device = args['device']

    if args['target_update_freq'] >= 0:
        target = deepcopy(model)
        target = target.to(model.device)
    else:
        target = None

    memory = ReplayMemoryOptimized(args['replay_size'],args['GAMMA'],args['H']*args['frame_stack'],args['num_channels'])

    train_reward_lst = deque(maxlen=500)
    for _ in range(50):
        train_reward_lst.append(0)


    for i_tran in range(N_traj):
        state = (torch.FloatTensor(env.reset()).to(model.device) / 255.)[:,:,:args['frame_stack']*args['num_channels']]
        traj_dict = defaultdict(list)
        traj_reward = 0
        done = False
        # logger.log(str(i_tran))
        for t in range(T_max):
            if done:
                break
            action = model.select_action(state)
            model.steps_done += 1
            
            next_state, reward, done, _ = env.step(action)
            traj_reward += reward
            reward = torch.FloatTensor([reward]).to(model.device)
            next_state = (torch.FloatTensor(next_state).to(model.device) / 255.) [:,:,:args['frame_stack']*args['num_channels']]
            # Observe new state
                
            # Store the transition in memory
            memory.append(state, action, reward, done)

            # Move to the next state
            state = next_state
            optimal_Ns = np.array([1.])

            if len(memory) >= args['warmup_steps']:
                # print("update")
                loss_dict = model.update(target, memory)
            else:
                loss_dict = {}

            for k,v in loss_dict.items():
                traj_dict[k].append(v)
            
            if done:
                break
        avg_metrics = defaultdict(float)
        loss_str = ""
        for k,v in loss_dict.items():
            avg_metrics[k] = np.mean(traj_dict[k])
            loss_str += " %s: %.5f " % (k,avg_metrics[k])

        if i_tran % args['test_interval'] == 0:
            eval_returns = eval_policy(args['test_episodes'], env, model, T_max, args)
            logger.add_scalar("eval_returns", eval_returns, i_tran)
        
        print('[Episode %d] %s returns: %.2f' % (i_tran,loss_str,traj_reward))
        logger.add_scalar("loss_rl", avg_metrics['loss_rl'], i_tran)
        logger.add_scalar("loss_infomax", avg_metrics['loss_infomax'], i_tran)
        logger.add_scalar("loss_vq", avg_metrics['vq_loss'], i_tran)
        train_reward_lst.append(traj_reward)
        rewards = list(train_reward_lst)
        mean100 = np.mean(rewards[-100:])
        # smooth the training rewards for 100 episodes
        logger.add_scalar("train_returns", mean100, i_tran)

        if target is not None and model.steps_done % args['target_update_freq'] == 0:
            soft_update(model, target, args['tau'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', type=str, default='FourRooms')
    parser.add_argument('--seed', type=int, default=0)
    # InfoMax arguments
    parser.add_argument('--lambda_LL', type=float, default=0)
    parser.add_argument('--lambda_LG', type=float, default=0)
    parser.add_argument('--lambda_GL', type=float, default=0)
    parser.add_argument('--lambda_GG', type=float, default=0)
    parser.add_argument('--lambda_rl', type=float, default=1)
    
    args = parser.parse_args()
    main(vars(args))
