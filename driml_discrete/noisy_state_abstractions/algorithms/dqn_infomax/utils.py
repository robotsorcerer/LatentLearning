import numpy as np
import random
import torch
from collections import defaultdict
import copy

from torch.utils.data import RandomSampler, BatchSampler

"""
Helper functions
- Convolution output calculator
"""

def compute_network_output_size(h,w,kernels_h,kernels_w,strides_h,strides_w):
    for (k_h,k_w,s_h,s_w) in zip(kernels_h,kernels_w,strides_h,strides_w):
        h = (h-k_h) / s_h + 1
        w = (w-k_w) / s_w + 1
    return int(h) * int(w)

def get_layers_grad(model,layer_list,flatten=True):
    acc = []
    acc2 = []
    acc3 = []
    for name,param in model.named_parameters():
        for layer in layer_list:
            if layer in name and param.grad is not None: # found match AND has grad
                if flatten:
                    grad = param.grad.view(-1).clone().detach()
                else:
                    grad = param.grad.clone().detach()
                acc.append( grad )
                acc2.append(param)
                acc3.append(name)
    return acc, acc2, acc3

def partition_list_by_lists(data,groups):
    partition = [[] for _ in range(len(groups)+1)]
    for el in data:
        found = False
        for g,group in enumerate(groups):
            for kw in group:
                if kw in el:
                        partition[g].append(el)
                        found = True
        if not found:
                partition[-1].append(el)
    return partition

def compute_model_grad_norm(model):
    grad_norm = 0
    for i,p in enumerate(model.parameters()):
        if p.grad is not None:
                grad_norm += p.grad.data.norm()
    return grad_norm / (i+1)


def filter_dict_by_dict(source,query):
    if not len(source):
        return False
    for k_query,v_query in query.items():
        if k_query in source:
            try:
                src = float(source[k_query])
            except:
                src = str(source[k_query])
            try:
                query = float(v_query)
            except:
                query = str(v_query)
            if src != query:
                return False
        else:
            return False
    return True

def init(module, weight_init, bias_init):
    weight_init(module.weight.data)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def make_one_hot(labels, C=2):

    one_hot = torch.FloatTensor(size=(labels.size(0),C)).zero_()
    if torch.cuda.is_available():
            one_hot = one_hot.cuda()
    target = one_hot.scatter_(1, labels.unsqueeze(-1).long(), 1).float()
    return target

def set_seed(seed,cuda):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def check_atari(env_name):
    atari_list = ap.list_games()
    env_name = env_name.lower()
    for game in atari_list:
        game = ''.join(game.split('_'))
        if game in env_name:
                return True
    return False

def navigation_alphabet():
    return {
            '#':{'reward_pdf':lambda :0,'terminal':False,'accessible':False,'color':[0,0,0],'stochastic':False,'initial':False,'collectible':False},
            ' ':{'reward_pdf':lambda :0,'terminal':False,'accessible':True,'color':[255,255,255],'stochastic':False,'initial':False,'collectible':False},
            'S':{'reward_pdf':lambda :0,'terminal':False,'accessible':True,'color':[255,255,255],'stochastic':False,'initial':True,'collectible':False},
            '0':{'reward_pdf':lambda :1,'terminal':True,'accessible':True,'color':[50,50,255],'stochastic':False,'initial':False,'collectible':False},
            '.agent':{'color':[50,205,50]}
            }

def rollout(agent,env,n,max_steps,model,args):
    state_acc, action_acc, reward_acc = [],[],[]
    for i in range(n):
            state = env.reset()
            states = [state.copy()]
            actions = []
            rewards = []
            for t in range(max_steps):
                state = (torch.FloatTensor(state).to(model.device) / 255.)[:,:,:args['frame_stack']*args['num_channels']]
                
                action = agent(state)
                actions.append(action)
                next_state, reward, done, _ = env.step(action)

                state = next_state.copy()
                rewards.append(reward)
                states.append(state)
                if done:
                    break
            state_acc.append(states)
            action_acc.append(actions)
            reward_acc.append(rewards)
    return state_acc,reward_acc,action_acc

def synchronized_rollout(agent,envs,n,max_steps,eps=0):
        state_acc, action_acc, reward_acc = defaultdict(list), [], []
        for i in range(n):
                done = False
                states = defaultdict(list)
                for env_name,env in envs.items():
                        state = env.reset()
                        states[env_name] = [state]
                dm_state = states['dm'][-1]
                actions = []
                rewards = 0
                for t in range(max_steps):
                        dm_state = torch.FloatTensor(dm_state).unsqueeze(0)
                        if torch.cuda.is_available():
                                dm_state = dm_state.cuda()
                        q_vals = agent(dm_state)[0]['output_top']
                        u = np.random.uniform(size=1)
                        if u < eps:
                                action = np.random.randint(low=0,high=agent.classifier.num_outputs,size=1)[0]
                        else:
                                action = q_vals.max(1)[1].cpu().detach().numpy().item()
                        actions.append(action)
                        for env_name,env in envs.items():
                                next_state, reward, done, _ = env.step(action)
                                states[env_name].append(next_state)
                        dm_state = states['dm'][-1]
                        rewards += reward
                        if done:
                                break
                for env_name,env in envs.items():
                        state_acc[env_name].append(states[env_name])
                action_acc.append(actions)
                reward_acc.append(rewards)
        return state_acc,reward_acc,action_acc

def rollout_TD(agent,env,max_steps,eps=0):
        states, actions, rewards = [],[],[]
        done = False
        state = env.reset()
        states = [state]
        t = 0
        while t < max_steps:
                state = torch.FloatTensor(state).unsqueeze(0)
                if torch.cuda.is_available():
                        state = state.cuda()
                q_vals = agent(state)
                u = np.random.uniform(size=1)
                if u < eps:
                        action = np.random.randint(low=0,high=agent.classifier.num_outputs,size=1)[0]
                else:
                        action = q_vals.max(1)[1].cpu().detach().numpy().item()
                actions.append(action)
                next_state, reward, done, _ = env.step(action)
                state = next_state

                rewards.append(reward)
                actions.append(actions)
                states.append(state)
                if done:
                        break
                t += 1
        return states,rewards,actions

def ram_rollout(agent,env,n,max_steps,reset_to_state=None):
        state_acc, action_acc, reward_acc, ram_acc = [],[],[],[]
        obs_type = copy.deepcopy( env.observation_type )
        for i in range(n):
                state = env.reset()
                env.unwrapped.observation_type = 'full_vector'
                ram_state = env.getCurrentObservation()
                env.unwrapped.observation_type = obs_type
                states = [state.copy()]
                ram_states = [ram_state]
                actions = []
                rewards = []
                for t in range(max_steps):
                        state = torch.FloatTensor(state)
                        if torch.cuda.is_available():
                                state = state.cuda()
                        
                        action = agent(state)
                        actions.append(action)
                        next_state, reward, done, info = env.step(action)

                        next_state = next_state.copy()

                        env.unwrapped.observation_type = 'full_vector'
                        ram_state = env.getCurrentObservation()
                        env.unwrapped.observation_type = obs_type

                        state = next_state
                        rewards.append(reward)
                        states.append(state)
                        ram_states.append(ram_state)
                        if done:
                                break
                state_acc.append(states)
                action_acc.append(actions)
                reward_acc.append(rewards)
                ram_acc.append(ram_states)
        return state_acc,reward_acc,action_acc,ram_acc

def make_batch(states,rewards,actions,ram,batch_size,device):
    total_steps = sum([len(e) for e in states])
    sampler = BatchSampler(RandomSampler(range(len(states)),
                                            replacement=True, num_samples=total_steps),
                            batch_size, drop_last=True)
    for indices in sampler:
        states_batch = [states[x] for x in indices]
        rewards_batch = [rewards[x] for x in indices]
        actions_batch = [actions[x] for x in indices]
        ram_batch = [ram[x] for x in indices]
        s_t, s_tp1, r_t, a_t, ram_t  = [], [], [], [], []
        for states_,rewards_,actions_,ram_ in zip(states_batch,rewards_batch,actions_batch,ram_batch):
            t = np.random.randint(0, len(states_)-1)

            s_t.append(states_[t])
            s_tp1.append(states_[t+1])
            r_t.append(rewards_[t])
            a_t.append(actions_[t])
            ram_t.append(ram_[t])
        yield torch.FloatTensor(s_t).to(device).permute(0,3,1,2) / 255., torch.FloatTensor(ram_t).to(device), torch.FloatTensor(a_t).to(device), torch.FloatTensor(r_t).to(device), torch.FloatTensor(s_tp1).to(device).permute(0,3,1,2) / 255.

def mat2idx(shape,coords):
        n_rows, n_cols = shape
        return (coords[0]-1)*(n_cols-1) + (coords[1]-1)

def parse_optimal_policy(policy):
        d = {'↑':[-1, 0],
             '↓':[1, 0],
             '←':[0, -1],
             '→':[0, 1]}
        pi_star = {}
        for i,x in enumerate(policy.split('\n')):
                for j,el in enumerate(x.strip()):
                        if el in d:
                                pi_star["%d,%d"%(i,j) ] = d[el]
        return pi_star

def make_pixelworld(env_name):
        from RLDIM.wrappers import wrap_deepmind_pixelworld
        if env_name == "PixelWorld-Empty-5x5-v0":
                world_map = """
                #######
                #    0#
                #     #
                #     #
                #     #
                #S    #
                #######
                """
                optimal_policy =  """
                #######
                #→→→→0#
                #↑↑↑↑↑#
                #↑↑↑↑↑#
                #↑↑↑↑↑#
                #S↑↑↑↑#
                #######
                """
                P = np.zeros((25,25))
                for i in range(25):
                        for j in range(25):
                                if i > 4:
                                        if i == j+5:
                                                P[i,j] = 1
                                else:
                                        if i+1 == j:
                                                P[i,j] = 1
                P[4,:] = 0
                # P[4,4] = 1
                gamma = 0.99
                R = np.zeros(25)
                # R = np.ones(25) * -1
                R[4] = 1
                I = np.identity(25)
                Q = np.linalg.inv(I - gamma * P) @ R
                action_space = "2d_discrete"

        if env_name == "PixelWorld-FourRooms-11x11-v0":
                world_map = """\
                #############
                #     #     #
                #     #     #
                #  S     S  #
                #     #     #
                #     #     #
                ## ####  0  #
                #     ### ###
                #     #     #
                #  S  #     #
                #           #
                #     #     #
                #############
                """
                optimal_policy = """\
                #############
                #↓↓↓↓↓#↓↓↓↓↓#
                #↓↓↓↓↓#↓↓↓↓↓#
                #→→→→→→↓↓↓↓↓#
                #↓↓↑↑↑#↓↓↓↓↓#
                #→↓←↑↑#↓↓↓↓↓#
                ##↓####→→↓←←#
                #↓↓↓↓↓###↓###
                #↓↓↓↓↓#↓↓↓↓↓#
                #↓↓↓↓↓#→→→→←#
                #→→→→→→→→→→↓#
                #↑↑↑↑↑#→→→→↑#
                #############
                """
                action_space = "2d_discrete"
                env = pixel_world.PixelWorld(reward_mapping=navigation_alphabet(),
                                     world_map=world_map,
                                     from_string=True,
                                     state_type='image',
                                     actions=action_space,
                                     channels_first=False,
                                     randomize_goals=False)
                n = len(env.accessible_states)
                P = np.zeros((n,n))
                toXY = {}
                toID = {}
                for i,state in enumerate(env.accessible_states):
                        toXY[i] = state.coords
                        toID[ "%d,%d"%(state.coords[0],state.coords[1]) ] = i
                opt = parse_optimal_policy(optimal_policy)
                for ID in range(n):
                        xy = toXY[ID]
                        coord = "%d,%d"%(xy[0],xy[1])
                        opt_a = opt[coord]
                        xy_next = xy + np.array(opt_a)
                        coord_next = "%d,%d"%(xy_next[0],xy_next[1])
                        # print(xy,xy_next)
                        ID_next = toID[coord_next]
                        P[ID,ID_next] = 1.
                goal = env.goal_states[0]
                ID_goal = toID["%d,%d"%(goal.coords[0],goal.coords[1])]

                gamma = 0.99
                R = 0 * np.ones(n)
                R[ID_goal] = 1.
                P[ID_goal,:] = 0.
                I = np.identity(n)
                Q = np.linalg.inv(I - gamma * P) @ R

                Q_mat = np.zeros((13,13))
                for ID in range(n):
                        xy = toXY[ID]
                        Q_mat[xy[0],xy[1]] = Q[ID]
                Q = Q_mat
        wrapper = wrap_deepmind_pixelworld
        env_maker = lambda x:(wrapper(pixel_world.PixelWorld(reward_mapping=navigation_alphabet(),
                                world_map=world_map,
                                from_string=True,
                                state_type='image',
                                actions=action_space,
                                channels_first=False)))
        env = pixel_world.PixelWorld(reward_mapping=navigation_alphabet(),
                                     world_map=world_map,
                                     from_string=True,
                                     state_type='image',
                                     actions=action_space,
                                     channels_first=False)
        env_xy = pixel_world.PixelWorld(reward_mapping=navigation_alphabet(),
                                     world_map=world_map,
                                     from_string=True,
                                     state_type='xy',
                                     actions=action_space,
                                     channels_first=False)
        return env_maker,env,env_xy, Q

if __name__ == "__main__":
        world_map = """\
                #############
                #     #     #
                #     #     #
                #  S        #
                #     #     #
                #     #     #
                ## ####  0  #
                #     ### ###
                #     #     #
                #     #     #
                #           #
                #     #     #
                #############
                """
        optimal_policy = """\
                #############
                #↓↓↓↓↓#↓↓↓↓↓#
                #↓↓↓↓↓#↓↓↓↓↓#
                #→→→→→→↓↓↓↓↓#
                #↑↑↑↑↑#↓↓↓↓↓#
                #↑↑↑↑↑#↓↓↓↓↓#
                ##↑####→→↑←←#
                #↓↓↓↓↓###↑###
                #↓↓↓↓↓#→→↑←←#
                #↓↓↓↓↓#↑↑↑↑↑#
                #→→→→→→↑↑↑↑↑#
                #↑↑↑↑↑#↑↑↑↑↑#
                #############
                """
        def navigation_alphabet():
                return {
                        '#':{'reward_pdf':lambda :0,'terminal':False,'accessible':False,'color':[0,0,0],'stochastic':False,'initial':False,'collectible':False},
                        ' ':{'reward_pdf':lambda :0,'terminal':False,'accessible':True,'color':[255,255,255],'stochastic':False,'initial':False,'collectible':False},
                        'S':{'reward_pdf':lambda :0,'terminal':False,'accessible':True,'color':[255,255,255],'stochastic':False,'initial':True,'collectible':False},
                        '0':{'reward_pdf':lambda :1,'terminal':True,'accessible':True,'color':[50,50,255],'stochastic':False,'initial':False,'collectible':False},
                        '.agent':{'color':[50,205,50]}
                        }
        env = pixel_world.PixelWorld(reward_mapping=navigation_alphabet(),
                                     world_map=world_map,
                                     from_string=True,
                                     state_type='image',
                                     actions='2d_discrete',
                                     channels_first=False,
                                     randomize_goals=True)
        n = len(env.accessible_states)
        P = np.zeros((n,n))
        toXY = {}
        toID = {}
        for i,state in enumerate(env.accessible_states):
                toXY[i] = state.coords
                toID[ "%d,%d"%(state.coords[0],state.coords[1]) ] = i
        opt = parse_optimal_policy(optimal_policy)
        for ID in range(n):
                xy = toXY[ID]
                coord = "%d,%d"%(xy[0],xy[1])
                opt_a = opt[coord]
                xy_next = xy + np.array(opt_a)
                coord_next = "%d,%d"%(xy_next[0],xy_next[1])
                # print(xy,xy_next)
                ID_next = toID[coord_next]
                P[ID,ID_next] = 1.
        goal = env.goal_states[0]
        ID_goal = toID["%d,%d"%(goal.coords[0],goal.coords[1])]

        gamma = 0.99
        R = 0 * np.ones(n)
        R[ID_goal] = 1.
        P[ID_goal,:] = 0.
        I = np.identity(n)
        Q = np.linalg.inv(I - gamma * P) @ R

        Q_mat = np.zeros((13,13))
        for ID in range(n):
                xy = toXY[ID]
                Q_mat[xy[0],xy[1]] = Q[ID]
        Q = Q_mat


        print(env.goal_states)
        x = env.reset()
        print(env.current_state.coords)
        #import matplotlib.pyplot as plt
        #plt.imshow(Q_mat,cmap='gray',vmin=0.8,vmax=1.0)
        #plt.colorbar()
        #plt.show()

        #exit()
        s,r,done, _ = env.step(1)
        print(r)
        #plt.imshow(s)
        #plt.show()
        print(env.current_state.coords)
        s,r,done, _ = env.step(0)
        print(r)
        print(env.current_state.coords)
        #plt.imshow(s)
        #plt.show()
