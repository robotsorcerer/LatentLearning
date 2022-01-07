# from comet_ml import OfflineExperiment

from RLDIM.losses import InfoNCE_action_loss
from RLDIM.utils import make_one_hot, set_seed
from RLDIM.models import *
from RLDIM.loggers import Logger
from RLDIM.wrappers import DistributionShiftWrapper, HistoryWrapper, WarpFramePocMan, IsingWrapper

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import namedtuple
from copy import deepcopy

import matplotlib.pyplot as plt
import cv2
import os

import uuid

import gym
import gym_minigrid
from gym_minigrid.wrappers import  FullyObsWrapper,RGBImgObsWrapper
import pocman_gym
from buffer import MultiEnvReplayMemory as ReplayMemory

import argparse
from scipy.special import softmax
from utils import rollout


"""
Wrappers
"""

class ClipAction(gym.ActionWrapper):
    r"""Clip the continuous action within the valid bound. """
    def __init__(self, env):
        super(ClipAction, self).__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        return action


class DQN(nn.Module):
        
        def __init__(self,input_space,output_space,device,EPS_START,EPS_END,EPS_DECAY,encoder='infoNCE_Mnih_84x84_action_FILM'):
            super().__init__()
            
            self.input_space = input_space
            self.output_space = output_space
            self.EPS_END = EPS_END
            self.EPS_START = EPS_START
            self.EPS_DECAY = EPS_DECAY
            
            num_outputs = output_space.n
            self.encoder = globals()[encoder](input_space, num_outputs, {})
            self.head = nn.Linear(512,num_outputs)
            
            self.steps_done = 0
            self.device = device
            
        def forward(self,x):
            x_relu = self.encoder(x)
            x = self.head(x_relu)
            return x
        
        def select_action(self,state):
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                math.exp(-1. * self.steps_done / self.EPS_DECAY)
            if sample > eps_threshold:
                with torch.no_grad():
                    q_vals = self.forward(state.unsqueeze(0).permute(0,3,1,2).contiguous())
                    return q_vals.max(1)[1].view(1, 1).squeeze(0).squeeze(0).item()
            else:
                return self.output_space.sample()

class Action_net(nn.Module):
    def __init__(self,n_actions):
        super(Action_net,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_actions, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        return self.layers(x)

def dqn_loss(model, target, states,actions,returns,next_states,nonterminals, gamma,batch_size):
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s ==1,nonterminals)), dtype=torch.bool).to(model.device)
        non_final_next_states = torch.stack([next_states[s] for s in range(len(next_states)) if non_final_mask[s] == 1])
        # state_batch = torch.cat(states).float()
        action_batch = actions.unsqueeze(1).to(model.device)
        # reward_batch = torch.cat(returns)
        state_action_values = model(states).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size).to(model.device)
        with torch.no_grad():
            if target is not None:
                _, next_state_actions = model(non_final_next_states).max(1, keepdim=True)
                next_state_values[non_final_mask] = target(non_final_next_states).gather(1, next_state_actions)[:,0].detach()
            else:
                next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * gamma) + returns

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        return loss

def adaptive_k(T,A_mat,actions,action_net,action_net_optimizer,device):
    action_type = 'hidden'
    N_hidden = A_mat.shape[0] 
    N_visible = A_mat.shape[0]
    N_actions = N_hidden

    def loss_fn(p,q):
        log_2 = math.log(2.)
        # return -F.log_softmax(p).mean()
        return (log_2 - F.softplus(- p)).mean() - (F.softplus(-q) + q - log_2).mean()

    B = len(actions[0])
    T = len(actions)
    Y = torch.ones(size=(B,)).to(device)
    C = torch.zeros(size=(B,)).to(device)

    for tt in range(1,T):
        """
        A) Estimate K for each entry
        """
        aa = actions[tt-1]
        bb = actions[tt]
        a_t = aa
        a_tp1 = bb

        P = torch.FloatTensor( np.array([A_mat[a_t[i],a_tp1[i] ] for i in range(B)])  ).to(device)
        S = torch.bernoulli(P)
        Y = Y * S
        C = C + Y

    """
    B) Update A_mat
    """
    
    true_inp = torch.cat([F.one_hot(actions[:-1].view(-1).long(),N_actions),
                          F.one_hot(actions[1:].view(-1).long(),N_actions)],1).float().to(device)
    p = action_net( true_inp )
    
    idx = torch.randperm(B*(T-1))
    perm_inp = torch.cat([F.one_hot(actions[:-1].view(-1).long(),N_actions),
                          F.one_hot(actions[1:].view(-1).long(),N_actions)[idx]],1).float().to(device)
    q = action_net( perm_inp )
    
    loss_ = -loss_fn(p,q)
    action_net_optimizer.zero_grad()
    loss_.backward()
    action_net_optimizer.step()

    """
    C) Do EMA on A_mat
    """
    
    A_hat = np.zeros(shape=(N_actions,N_actions))
    v1 = F.one_hot(torch.arange(N_actions).repeat_interleave(N_actions),N_actions)
    v2 = F.one_hot(torch.arange(N_actions).repeat(N_actions),N_actions)
    v = torch.cat([v1,v2],dim=1).float().to(device)
    A_hat = torch.softmax(action_net(v).detach().cpu().view(N_actions,N_actions),dim=1).numpy()

    
    A_mat = softmax((0.9) * A_mat + (0.1) * A_hat,1)


    optimal_Ns = np.maximum(1,C.detach().cpu().numpy())

    adx = torch.arange(0, len(actions[0])).long()

    average_N = optimal_Ns.mean()

    return loss_, optimal_Ns, adx, A_mat

def soft_update(model, target, tau):
        for target_param, local_param in zip(target.parameters(), model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def get_state_metric(rewards,env):
    current_env_idx = env.current_env_idx
    N_envs = len(env.envs)
    current_indices = ( np.arange(0,len(rewards)) % N_envs ) == current_env_idx
    other_indices = ( np.arange(0,len(rewards)) % N_envs ) != current_env_idx
    
    current_env_rewards = np.mean([sum(rewards[i]) for i in range(len(rewards)) if current_indices[i]])
    other_env_rewards = np.mean([sum(rewards[i]) for i in range(len(rewards)) if other_indices[i]])

    return current_env_rewards, other_env_rewards
    

def main(cmd_args):
    env_name = 'PocManFullRGB-v0'
    seed = int(np.random.randint(0,100000000))
    # set_seed(seed,torch.cuda.is_available())

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


    args = {'env_name':env_name,
            'device':'cuda:0' if torch.cuda.is_available() else 'cpu',
            'seed':seed,
            'score_fn':'nce_scores_log_softmax',
            'lambda_LL':cmd_args['lambda_LL'],
            'lambda_GL':cmd_args['lambda_GL'],
            'lambda_LG':cmd_args['lambda_LG'],
            'lambda_GG':cmd_args['lambda_GG'],
            'lambda_rl':cmd_args['lambda_rl'], 
            'BATCH_SIZE':32,
            'GAMMA':0.95,
            'N_traj':22000,
            'warmup_steps':1000,
            'T_max':200,
            'replay_size':500000,
            'frame_stack':4,
            'num_channels':3,
            'EPS_START':1.,
            'EPS_END':0.1,
            'EPS_DECAY':10000,
            'target_update_freq':-1,
            'tau':5e-3,
            'weight_save_interval':100,
            'test_interval':1000,
            'test_episodes':1,
            'data_aug':0,
            'eps_list':cmd_args['eps_list'],
            'ema_moco':0,
            'ising':True,
            'H':5,
            'k':-1
            }

    weight_save_window = 50
    list_of_eps = cmd_args['eps_list']
    list_of_eps = np.array([list(map(float,x.split('->'))) for x in list_of_eps.split(',')]).transpose().tolist()
    
    env_train_kwargs = {
                'id':env_name,
                'harmless_ghosts' : [ [i for i in range(4) if i!=j] for j in range(4) ], # [i for i in range(4) if i!=j] for j in range(2)
                'wall_place_prob': [0.0],
                'food_place_prob':[0.2],
                'ghost_random_move_prob':list_of_eps
                } 
    env_test_kwargs = {
                'id':env_name,
                'harmless_ghosts' : [ [i for i in range(4) if i!=j] for j in range(4) ], # [i for i in range(4) if i!=j] for j in range(2)
                'wall_place_prob': [0.0],
                'food_place_prob':[0.2],
                'ghost_random_move_prob':list_of_eps
                } 

    args = {**args,**env_train_kwargs}

    env_maker = lambda env_args:HistoryWrapper( IsingWrapper(WarpFramePocMan( gym.make(**env_args) ,
                                                        21*2,19*2,grayscale=(args['num_channels'] == 1)),21*2,19*2),X=args['frame_stack']*args['H'])
    episodes_per_env_train = 5000
    episodes_per_env_test = 1
    env = DistributionShiftWrapper( env_maker, episodes_per_env=episodes_per_env_train,kwargs=env_train_kwargs)
    env_test = DistributionShiftWrapper( env_maker, episodes_per_env=episodes_per_env_test,kwargs=env_test_kwargs)
    N_envs_train = len(env.envs)
    args['replay_size'] = args['replay_size'] // N_envs_train
    

    exp_name = 'DQN'+ ('_vanilla' if args['lambda_LL']==args['lambda_LG']==args['lambda_GL']==args['lambda_GG']==0 else '_nce')

    exp_path = os.path.join('pacman_ising_ablation', exp_name+str(seed) )

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    if not os.path.isdir('A_mats'):
        os.makedirs('A_mats')

    logger = Logger(exp_path, use_TFX=False, params=args)

    # env = ClipAction(WarpFrame(RGBImgObsWrapper(gym.make('MiniGrid-KeyCorridorS3R1-v0')),45,45,grayscale=False))
    # env = ClipAction(WarpFrame(RGBImgObsWrapper(gym.make('MiniGrid-DistShift1-v0')),45,45,grayscale=False))

    """
    To run:

    NVIDIA_VIDIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python nce_in_gridworld.py


    Dist shift:

    Observation space: 84x84x3
    Action space: 0,1,2,..,6
    """

    """
    PocMan:

    Observation space: 21 x 19 x 3 --> 42 x 38 x 3
    Action space: 0,1,2,3
    """

    BATCH_SIZE = args['BATCH_SIZE']
    GAMMA = args['GAMMA']
    N_traj = args['N_traj']
    T_max = args['T_max']
    EPS_START = args['EPS_START']
    EPS_END = args['EPS_END']
    EPS_DECAY = args['EPS_DECAY']
    weight_save_interval = args['weight_save_interval'] # save every X episodes

    model = DQN(np.zeros(shape=(env.observation_space.shape[0],env.observation_space.shape[1],args['frame_stack']*args['num_channels'])),env.action_space,device,EPS_START,EPS_END,EPS_DECAY)
    model = model.to(model.device)
    N_actions = env.action_space.n
    A_mat = np.random.uniform(size=(N_actions,N_actions))
    action_net = Action_net(N_actions*2)
    action_net = action_net.to(model.device)
    action_net_optimizer = optim.Adam(action_net.parameters(),lr=1e-4)

    device = args['device']
    
    # tau = Tau(512)
    # tau_t = Tau(512)
    # vee = torch.rand(1, requires_grad=True ,device=device)
    # if device == 'cuda':
    #     tau = tau.cuda()
    #     tau_t = tau_t.cuda()

    # tau_t.load_state_dict(tau.state_dict())

    # opt_tau = torch.optim.Adam(tau.parameters(),lr=0.01)
    # opt_vee = torch.optim.Adam([vee],lr=0.01)

    if args['target_update_freq'] >= 0:
        target = deepcopy(model)
        target = target.to(model.device)
    else:
        target = None
    opt = optim.Adam(model.parameters(),lr=2.5e-4)
    memory = ReplayMemory(args['replay_size'],GAMMA,args['H']*args['frame_stack'],args['num_channels'],N_envs_train)

    # psi_t = model.encoder.psi_global_GG_t
    # psi_tp1 = model.encoder.psi_global_GG_t_p_1

    def agent(state):
        return model.select_action(state)

    states,rewards,actions= rollout(agent,env_test,N_envs_train * args['test_episodes'],T_max,model,args)
    avg_reward_test_current,avg_reward_test_others  = get_state_metric(rewards,env_test)
    # logger.log( {'name':'reward_test_current', 'value':avg_reward_test_current, 'step':0 })
    # logger.log( {'name':'reward_test_others', 'value':avg_reward_test_others, 'step':0 })

    # experiment = OfflineExperiment(offline_directory="pacman_ising_ablation/",auto_output_logging=False,project_name='nce-procgen',workspace="bmazoure",disabled=False)
    # experiment.add_tag('PacMan')
    # experiment.set_name('PacMan_ablation')
    # experiment.log_parameters(args)

    for i_tran in range(N_traj):
        state = (torch.FloatTensor(env.reset()).to(model.device) / 255.)[:,:,:args['frame_stack']*args['num_channels']]
        traj_loss_nce = []
        traj_loss_rl = []
        traj_reward = 0
        done = False
        logger.log(str(i_tran))
        for t in range(T_max):
            if done:
                break
            action = model.select_action(state)
            model.steps_done += 1
            # action = [2,1,2,2,0,2,2,2,2,2,0,2,2][t]
            next_state, reward, done, _ = env.step(action)
            traj_reward += reward
            reward = torch.FloatTensor([reward]).to(model.device)
            next_state = (torch.FloatTensor(next_state).to(model.device) / 255.) [:,:,:args['frame_stack']*args['num_channels']]
            # Observe new state
                

            # Store the transition in memory
            memory.append(state, action, reward, done, env.current_env_idx)
            # memory.push(state.long(), action, next_state, reward)

            # Move to the next state
            state = next_state
            optimal_Ns = np.array([1.])

            if len(memory.replays[env.current_env_idx]) >= args['warmup_steps']:# and (model.steps_done % args['frame_stack']) ==0:

                tree_idxs, states, actions, returns, next_states, nonterminals, weights = memory.sample(BATCH_SIZE,env.current_env_idx)
                actions=F.relu(actions)
                a_t = actions[:,2] # for loss RL
                
                loss_rl = dqn_loss(model, target, states[:,:args['num_channels']*args['frame_stack']] ,a_t,returns,next_states[:,:args['num_channels']*args['frame_stack']] ,nonterminals, GAMMA,BATCH_SIZE)
                
                if args['lambda_LL'] >0 or args['lambda_LG'] >0 or args['lambda_GL'] >0 or args['lambda_GG'] > 0:
                    
                    

                    """
                    NCE loss
                    """
                    non_final_mask = torch.tensor(tuple(map(lambda s: s ==1,nonterminals)), dtype=torch.bool).to(model.device)
                    a_t = a_t[non_final_mask]
                    s_t = states[non_final_mask][:,:args['num_channels']*args['frame_stack']]
                    r_t = returns[non_final_mask]
                    s_tp1 = states[non_final_mask][:,:args['num_channels']*args['frame_stack']]

                    """
                    Adaptive k
                    """
                    if args['k'] == '-1':
                        loss_, optimal_Ns, adx, A_mat = adaptive_k(args['H'],A_mat,actions.transpose(1,0).contiguous(),action_net,action_net_optimizer,device)
                        adx = torch.arange(0, len(actions)).long()
                        acc = []
                        for c in range(args['num_channels']*args['frame_stack']):
                            acc.append( states[adx[None,:],optimal_Ns+c][0] )
                            
                        next_states2 = torch.stack(acc).transpose(1,0)
                        s_tp1 = torch.stack([next_states2[s] for s in range(len(next_states2)) if non_final_mask[s] == 1])
                    
                    actions = actions.float().to(model.device)
                    
                    
                    dict_nce = InfoNCE_action_loss(model.encoder,s_t,a_t,r_t,s_tp1,args)
                    
                    nce_scores = args['lambda_LL'] * dict_nce['nce_L_L'] + \
                            args['lambda_LG'] * dict_nce['nce_L_G'] + \
                            args['lambda_GL'] * dict_nce['nce_G_L'] + \
                            args['lambda_GG'] * dict_nce['nce_G_G']
                    
                    loss_nce = -nce_scores.sum(0).mean()
                else:
                    loss_nce = torch.FloatTensor([0.]).to(model.device)
                    optimal_Ns = np.array([1.])
            
                loss = loss_nce + args['lambda_rl'] * loss_rl

                L_NCE = loss_nce.detach().cpu().item()
                L_RL = loss_rl.detach().cpu().item()

                # Optimize the model
                opt.zero_grad()
                loss.backward()
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
                opt.step()
            else:
                L_NCE = 0.
                L_RL = 0.

            traj_loss_rl.append(L_RL)
            traj_loss_nce.append(L_NCE)
            
            if done:
                break

        ep_loss_nce = np.mean(traj_loss_nce)
        ep_loss_rl = np.mean(traj_loss_rl)
        print('[Episode %d] Loss RL: %.5f Loss NCE_mean: %.5f Loss NCE_max: %.5f average N: %.2f CumReward: %.2f' % (i_tran,ep_loss_rl,ep_loss_nce,np.max(traj_loss_nce),optimal_Ns.mean(),traj_reward))
        logger.log({'name':'Average_N','value':optimal_Ns.mean(),'step':i_tran})
        logger.log({'name':'loss_rl','value':ep_loss_rl,'step':i_tran})
        logger.log({'name':'loss_nce_mean','value':ep_loss_nce,'step':i_tran})
        logger.log({'name':'loss_nce_max','value':np.max(traj_loss_nce),'step':i_tran})
        logger.log({'name':'reward_train','value':traj_reward,'step':i_tran})

        if (weight_save_interval > 0) and (i_tran %weight_save_interval) == 0:
            np.save("A_mats/%d.npy"%i_tran,A_mat)

        if weight_save_interval > 0 and ( i_tran % weight_save_interval ) < weight_save_window :
            if ( i_tran % weight_save_interval ) == 0:
                current_best_reward = float('-inf')
                episode_nb = i_tran
            if current_best_reward < traj_reward:
                # weight_path = os.path.join(exp_path,'tau_weights_%d.pth'%episode_nb)
                # torch.save(tau.state_dict(), weight_path)
                print('Saved tau weights at episode %d'%episode_nb)
                current_best_reward = traj_reward
        else:
            current_best_reward = float('-inf')

        if target is not None and model.steps_done % args['target_update_freq'] == 0:
            soft_update(model,target,args['tau'])
        
        if (i_tran % args['test_interval']) == 0:
            def agent(state):
                return model.select_action(state)
            states,rewards,actions= rollout(agent,env_test,N_envs_train * args['test_episodes'],T_max,model,args)
            avg_reward_test_current,avg_reward_test_others  = get_state_metric(rewards,env_test)
            logger.log({'name':'reward_test_current','value':avg_reward_test_current,'step':i_tran})
            logger.log({'name':'reward_test_others','value':avg_reward_test_others,'step':i_tran})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--lambda_LL', type=float, default=0)
    parser.add_argument('--lambda_LG', type=float, default=0)
    parser.add_argument('--lambda_GL', type=float, default=0)
    parser.add_argument('--lambda_GG', type=float, default=0)
    parser.add_argument('--lambda_rl', type=float, default=1)
    parser.add_argument('--eps_list', type=str, default='0,0,0,0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(vars(args))
