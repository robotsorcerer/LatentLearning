from noisy_state_abstractions.algorithms.rl_agents import DQNAgent

from noisy_state_abstractions.algorithms.dqn_infomax.models import Action_net
from noisy_state_abstractions.algorithms.dqn_infomax.losses import InfoNCE_action_loss

import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from scipy.special import softmax

class InfoMaxAgent(DQNAgent):
    def __init__(self,input_space,output_space,device,EPS_START,EPS_END,EPS_DECAY,tau,batch_size,num_channels,frame_stack, gamma, 
                lambda_LL,
                lambda_LG,
                lambda_GL,
                lambda_GG,
                lambda_rl,
                predictive_k,
                encoder='infoNCE_Mnih_84x84_action_FILM'):
        super().__init__(input_space,output_space,device,EPS_START,EPS_END,EPS_DECAY,tau,batch_size,num_channels,frame_stack, gamma, encoder)

        self.lambda_LL = lambda_LL
        self.lambda_LG = lambda_LG
        self.lambda_GL = lambda_GL
        self.lambda_GG = lambda_GG
        self.lambda_rl = lambda_rl
        self.predictive_k = predictive_k
        
        N_actions = output_space.n
        self.A_mat = np.random.uniform(size=(N_actions,N_actions))
        self.action_net = Action_net(N_actions*2)
        self.action_net = self.action_net.to(device)
        self.action_net_optimizer = optim.Adam(self.action_net.parameters(),lr=1e-4,amsgrad=True)

    def update(self, target, memory):
        _, states, actions, returns, next_states, nonterminals, weights = memory.sample(self.batch_size)
        actions=F.relu(actions)
        idx = self.num_channels*self.frame_stack
        # h, w = states.shape[2:4]
        a_t = actions[:,idx-1]
        
        s_t = states[:,:idx]#.permute(0,1,4,2,3)
        s_tp1 = next_states[:,:idx]#.permute(0,1,4,2,3)#.reshape(self.batch_size,-1,h,w)
        s_tpk = states[:,self.predictive_k*idx:(self.predictive_k+1)*idx]#.permute(0,1,4,2,3).reshape(self.batch_size,-1,h,w)
        loss_rl = self.rl_loss(target, s_t ,a_t,returns, s_tp1 ,nonterminals, self.gamma,self.batch_size)
        if self.lambda_LL >0 or self.lambda_LG >0 or self.lambda_GL >0 or self.lambda_GG > 0:
            """
            InfoMax
            """
            non_final_mask = nonterminals.bool().to(self.device)[:,0]
            a_t = a_t[non_final_mask]
            s_t = s_t[non_final_mask]
            r_t = returns[non_final_mask]
            s_tpk = s_tpk[non_final_mask]

            """
            Adaptive k
            """
            H = memory.history // self.frame_stack
            
            actions = actions.float().to(self.device)
            
            
            dict_nce = InfoNCE_action_loss(self.encoder,s_t,a_t,r_t,s_tpk,self.device,{'data_aug':False,
                                                                            'ema_moco':False,
                                                                            'lambda_LL':self.lambda_LL,
                                                                            'lambda_LG':self.lambda_LG,
                                                                            'lambda_GL':self.lambda_GL,
                                                                            'lambda_GG':self.lambda_GG})
            
            nce_scores = self.lambda_LL * dict_nce['nce_L_L'] + \
                    self.lambda_LG * dict_nce['nce_L_G'] + \
                    self.lambda_GL * dict_nce['nce_G_L'] + \
                    self.lambda_GG * dict_nce['nce_G_G']
            
            loss_nce = -nce_scores.sum(0).mean()
        else:
            loss_nce = torch.FloatTensor([0.]).to(self.device)
            optimal_Ns = np.array([1.])

        
        loss = loss_nce + self.lambda_rl * loss_rl

        # Optimize the model
        self.opt.zero_grad()
        loss.backward()
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.opt.step()

        return {'loss_rl':loss_rl.cpu().detach(),'loss_infomax':loss_nce.cpu().detach()}