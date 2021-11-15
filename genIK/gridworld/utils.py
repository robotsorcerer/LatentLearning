import numpy as np
import random
import os
import time
import json
import torch

import numpy as np
import matplotlib.pyplot as plt
# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py


create_folder = lambda f: [ os.makedirs(f) if not os.path.exists(f) else False ]

class Logger(object):
      def __init__(self, args, experiment_name='', environment_name='', type_decoder='', obs_type = '', use_exo ='', groups = '', folder='./results'):
            """
            Original: Original implementation of the algorithms
            HDR: Used Qhat
            HDR_RG: Uses Qhat where graph is retained
            DR: Uses Qhat-Vhat
            DR_RG: Uses
            """
            self.rewards = []
              
            self.save_folder = os.path.join(folder, experiment_name, type_decoder, obs_type, use_exo, environment_name, groups, time.strftime('%y-%m-%d-%H-%M-%s'))

            create_folder(self.save_folder)
            self.returns_critic_loss = []
            self.returns_reward_loss = []
            self.returns_actor_loss = []


      def record_type1_errors(self, type1_error):
            self.type1_errors = type1_error

      def record_type2_errors(self, type2_error):
            self.type2_errors = type2_error




      def record_reward(self, reward_return):
            self.returns_eval = reward_return

      def training_record_reward(self, reward_return):
            self.returns_train = reward_return

      def record_critic_loss(self, critic_loss):
          self.returns_critic_loss.append(critic_loss)

      def record_reward_loss(self, reward_loss):
          self.returns_reward_loss.append(reward_loss)

      def record_actor_loss(self, actor_loss):
          self.returns_actor_loss.append(actor_loss)

      def save(self):
            # np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.returns_eval)
            np.save(os.path.join(self.save_folder, "type1_errors.npy"), self.type1_errors)
            np.save(os.path.join(self.save_folder, "type2_errors.npy"), self.type2_errors)

      # def save_2(self):
      #       np.save(os.path.join(self.save_folder, "returns_train.npy"), self.returns_train)

      # def save_critic_loss(self):
      #       np.save(os.path.join(self.save_folder, "critic_loss.npy"), self.returns_critic_loss)

      # def save_reward_loss(self):
      #       np.save(os.path.join(self.save_folder, "reward_loss.npy"), self.returns_reward_loss)

      # def save_actor_loss(self):
      #       np.save(os.path.join(self.save_folder, "actor_loss.npy"), self.returns_actor_loss)


      # def save_policy(self, policy):
      #     torch.save(policy.actor.state_dict(), '%s/actor.pth' % (self.save_folder))
      #     torch.save(policy.critic.state_dict(), '%s/critic.pth' % (self.save_folder))
            #policy.save(directory=self.save_folder)

      def save_args(self, args):
            """
            Save the command line arguments
            """
            with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
                  json.dump(dict(args._get_kwargs()), f)



def plot_state_visitation(x, y, save_folder, bins):
      # np.histogram2d(states[:,0], states[:,1], bins=6)

      gridx = np.linspace(min(x),max(x),bins)
      gridy = np.linspace(min(y),max(y),bins)

      H, xedges, yedges = np.histogram2d(x, y, bins=[gridx, gridy])
      plt.figure()
      plt.plot(x, y, 'ro')
      plt.grid(True)


      myextent  =[xedges[0],xedges[-1],yedges[0],yedges[-1]]
      plt.imshow(H.T,origin='low',extent=myextent,interpolation='nearest',aspect='auto')
      plt.plot(x,y,'ro')
      plt.colorbar()
      plt.savefig(save_folder + '/state_visitation.png')

