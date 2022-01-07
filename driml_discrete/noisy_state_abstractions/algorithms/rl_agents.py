import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import math

from noisy_state_abstractions.algorithms.dqn_infomax.models import *

class DQNAgent(nn.Module):
        
    def __init__(self,input_space,output_space,device,EPS_START,EPS_END,EPS_DECAY,tau,batch_size,num_channels,frame_stack, gamma, encoder='infoNCE_Mnih_84x84_action_FILM'):
        super().__init__()
        
        self.input_space = input_space
        self.output_space = output_space
        self.EPS_END = EPS_END
        self.EPS_START = EPS_START
        self.EPS_DECAY = EPS_DECAY
        self.tau = tau
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.frame_stack = frame_stack
        self.gamma = gamma

        num_outputs = output_space.n
        self.encoder = globals()[encoder](input_space, num_outputs, {})
        self.head = nn.Linear(512,num_outputs)
        
        self.steps_done = 0
        self.device = device

        self.opt = optim.Adam(self.parameters(), lr=2.5e-4, amsgrad=True)
        
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

    def select_greedy_action(self, state):
        with torch.no_grad():
            q_vals = self.forward(state.unsqueeze(0).permute(0,3,1,2).contiguous())
            return q_vals.max(1)[1].view(1, 1).squeeze(0).squeeze(0).item()

    def rl_loss(self, target, states,actions,returns,next_states,nonterminals, gamma,batch_size):
        non_final_mask = nonterminals.bool().to(self.device)[:,0]
        non_final_next_states = next_states[non_final_mask]
        action_batch = actions.unsqueeze(1).to(self.device)
        
        state_action_values = self.forward(states).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size).to(self.device)
        with torch.no_grad():
            if target is not None:
                _, next_state_actions = self.forward(non_final_next_states).max(1, keepdim=True)
                next_state_values[non_final_mask] = target(non_final_next_states).gather(1, next_state_actions)[:,0].detach()
            else:
                next_state_values[non_final_mask] = self.forward(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * gamma) + returns

        # print("current", state_action_values)
        # print("target", expected_state_action_values.unsqueeze(1))
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        return loss

    def update(self, target, memory):
        tree_idxs, states, actions, returns, next_states, nonterminals, weights = memory.sample(self.batch_size)
        actions=F.relu(actions)
        a_t = actions[:,0]
        states = states[:,0]
        next_states = next_states[:,0]
        loss_rl = self.rl_loss(target, states[:,:self.num_channels*self.frame_stack] ,a_t,returns,next_states[:,:self.num_channels*self.frame_stack] ,nonterminals, self.gamma,self.batch_size)

        # Optimize the model
        self.opt.zero_grad()
        loss_rl.backward()
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.opt.step()

    def soft_update(self, target):
        for target_param, local_param in zip(target.parameters(), self.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)