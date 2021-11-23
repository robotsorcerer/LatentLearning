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


      def record_abstraction_accuracy(self, abstraction_accuracy):
            self.abstraction_accuracy = abstraction_accuracy

      def record_abstraction_error(self, abstraction_error):
            self.abstraction_error = abstraction_error



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

            np.save(os.path.join(self.save_folder, "abs_err.npy"), self.abstraction_error)
            np.save(os.path.join(self.save_folder, "abs_acc.npy"), self.abstraction_accuracy)


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
      plt.close()



def plot_code_to_state_visualization(state_count_for_code, code_number, save_folder, n_embed, statetogrid, counter, gridsize):

      # for i in code_number : 

      i = counter
      data = state_count_for_code[ counter ]

      data_list = list(data.flatten())
      highest_state = max(data_list, key=data_list.count, default=0)
      highest_state_pos = statetogrid[highest_state]

      bins = np.arange(0, gridsize - 1, 1)
      plt.xlim([0, gridsize - 1])
      plt.ylim([0, n_embed-1])
      # plt.text(1,20,'Hello World !')
      plt.text(1, 20, 'Highest State - Position - ' + str(highest_state_pos) + '- ' + 'State Number - ' + str(highest_state))

      plt.hist(data, bins=bins, alpha=0.5)
      plt.title('States captured by Codebook Element - ' + str(i) )
      plt.xlabel('States (0 - 36)')
      plt.ylabel('State Count for given Codebook Element')
      plt.savefig(save_folder + '/codebook_visitation_' + str(i) + '.png')

      plt.close()




def get_eval_error(z0, z1, s0, s1):
    ## Type 1: DSM (different states merged)
    ## Type 2: SSS (same state separated)
    type1_err=0
    type2_err=0
    abs_acc=0
    abs_err=0

    s0 = torch.tensor(s0)
    s1 = torch.tensor(s1)

    for i in range(z0.shape[0]):
        Z = z0[i] == z1[i]
        Z = Z.long()
        Z_comp = torch.prod(Z)

        S = s0[i] == s1[i]
        S = S.long()
        S_comp = torch.prod(S)

        if Z_comp and (1-S_comp):
            # Error 1: Merging states which should not be merged
            type1_err += 1

        if (1-Z_comp) and S_comp :
            #Error 2: Did not merge states which should be merged
            type2_err += 1

        if Z_comp and S_comp : 
            abs_acc += 1

        if (1 - Z_comp) and (1 - S_comp):
            abs_err += 1

    type1_err = type1_err/z0.shape[0] * 100
    type2_err = type2_err/z0.shape[0] * 100
    abs_acc = abs_acc/z0.shape[0] * 100
    abs_err = abs_acc/z0.shape[0] * 100

    return type1_err, type2_err, abs_acc, abs_err



def states_pos_to_int(test_s0, test_s1, rows, cols):
    test_s0_positions = np.zeros(test_s0.shape[0])
    test_s1_positions = np.zeros(test_s1.shape[0])
    gridtostate = {}
    statetogrid = {}
    count = 0
    for y in range(rows):
        for x in range(cols):
            gridtostate[(x, y)] = count
            statetogrid[count] = (x, y)
            count += 1

    for i in range(test_s0.shape[0]):
        test_s0_positions[i] = gridtostate[tuple(test_s0[i])]
        test_s1_positions[i] = gridtostate[tuple(test_s1[i])]

    return test_s0_positions, test_s1_positions, statetogrid
