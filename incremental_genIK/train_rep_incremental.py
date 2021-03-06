import imageio
import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import seeding
import sys
import torch
from tqdm import tqdm
from models.geniknet import GenIKNet
from visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld, MazeWorld, SpiralWorld, LoopWorld
from visgrid.utils import get_parser
from visgrid.sensors import *
from det_tabular_mdp_builder import DetTabularMDPBuilder
from value_iteration import ValueIteration
from utils import Logger, plot_state_visitation, plot_code_to_state_visualization
import random
from matplotlib import pyplot as plt
from policies import StationaryStochasticPolicy
from utils import get_eval_error, states_pos_to_int, position_to_integer_convert
import statistics
from scipy import stats as statistics

parser = get_parser()

parser.add_argument('--type', type=str, default='genIK', choices=['genIK'],
                    help='Which type of representation learning method')

parser.add_argument('-n','--n_updates', type=int, default=3000,
                    help='Number of training updates')

parser.add_argument('-r','--rows', type=int, default=3,
                    help='Number of gridworld rows')

parser.add_argument('-c','--cols', type=int, default=3,
                    help='Number of gridworld columns')

parser.add_argument('-w', '--walls', type=str, default='empty', choices=['empty', 'maze', 'spiral', 'loop'],
                    help='The wall configuration mode of gridworld')

parser.add_argument('-l','--latent_dims', type=int, default=128,
                    help='Number of latent dimensions to use for representation')

parser.add_argument('--L_genik', type=float, default=0.0,
                    help='Coefficient multi-step inverse dynamics loss')

parser.add_argument('-lr','--learning_rate', type=float, default=0.003,
                    help='Learning rate for Adam optimizer')

parser.add_argument('--batch_size', type=int, default=50,
                    help='Mini batch size for training updates')

parser.add_argument('--n_test_samples', type=int, default=100,
                    help='Number of test samples to use for the representation')

parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')

parser.add_argument('-t','--tag', type=str, required=False,
                    help='Tag for identifying experiment')

parser.add_argument('--save', action='store_true',
                    help='Save final network weights')
# VQ Discrete Layer
parser.add_argument('--use_vq', action='store_true',
                    help='Use VQ layer after the phi network')

parser.add_argument('--groups', type=int, default=1,
                    help='No. of groups to use for VQ-VAE')

parser.add_argument('--n_embed', type=int, default=10,
                    help='No. of embeddings')
# Clustering layer
parser.add_argument("--use_logger", action="store_true", default=False, help='whether to use logging or not')

parser.add_argument("--folder", type=str, default='./results/')

parser.add_argument("--use_two_mazes", action="store_true", default=False, help='whether to use observations from multiple images')

args = parser.parse_args()
seeding.seed(args.seed)

#% ------------------ Define MDP ------------------
env = GridWorld(rows=args.rows, cols=args.cols)
env_2 = GridWorld(rows=args.rows, cols=args.cols)
env_name = 'gridworld'

#% ------------------ LOGGERS ------------------
if args.use_logger:    
    exogenous = 'exo_noise'
    obs_type = 'obs_map'    
    logger = Logger(args, experiment_name=args.tag, environment_name=env_name, type_decoder=args.type,  obs_type = obs_type, use_exo = exogenous, groups = 'groups_' + str(args.groups) + '_embed_' + str(args.n_embed), folder=args.folder)
    logger.save_args(args)
    print('Saving to', logger.save_folder)
else:
    logger = None

n_samples = args.n_test_samples
start_state = env.get_state()
states = [start_state]
actions = []

# horizon = args.n_test_samples
# model = DetTabularMDPBuilder(actions=env.actions, horizon=horizon, gamma=1.0)  
if args.use_two_mazes:
    start_state_env_2 = env_2.get_state()
    states_env_2 = [start_state_env_2]
    actions_env_2 = []

current_state = env.get_state()
curr_state, _  = position_to_integer_convert(current_state, current_state, args.rows, args.cols)


#% ------------------ Generate experiences ------------------
policy = StationaryStochasticPolicy(len(env.actions), obs_dim=current_state.shape[0])

# two env rollouts - one with policy \pi and another with random action
for step in range(0, n_samples, 1):
    a = np.random.choice(env.actions)
    next_state, reward, _ = env.step(a)
    # model.add_state(curr_state)
    curr_state, state_next  = position_to_integer_convert(current_state, next_state, args.rows, args.cols)

    states.append(next_state)
    actions.append(a)
    # model.add_state(state_next)
    # model.add_reward(curr_state, a, float(reward))
    # model.add_transition(curr_state, a, state_next)

    if args.use_two_mazes:
        a2 = policy.sample_action(current_state)  
        next_state2, reward2, _ = env_2.step(a2)
        states_env_2.append(next_state2)
        actions_env_2.append(a2)
<<<<<<< HEAD

        # model.add_state(state=tuple(next_state) , timestep=step)
        model.add_state(next_state, timestep=step)

        model.add_transition(tuple(current_state), a, tuple(next_state))

        model.add_reward(tuple(current_state-1), a, float(reward))
        current_state = next_state

print('states', model._states)
print('transitions', model._transitions)

# model.finalize()
value_it = ValueIteration()
q_val = value_it.do_value_iteration(tabular_mdp=model, min_reward_val=0.0,horizon=3)
=======
# # model.finalize()
# value_it = ValueIteration()
# q_val = value_it.do_value_iteration(tabular_mdp=model, horizon=horizon, min_reward_val=0.0)
>>>>>>> 18cff2aec28827e4802d964146bd2035a440e497
# expected_ret = q_val[(0, (0, 0))].max()

states = np.stack(states)
s0 = np.asarray(states[:-1, :])
s1 = np.asarray(states[1:, :])
a = np.asarray(actions)

if args.use_two_mazes:
    states_env_2 = np.stack(states_env_2)
    s0_env_2 = np.asarray(states_env_2[:-1, :])
    s1_env_2 = np.asarray(states_env_2[1:, :])
    a_env_2 = np.asarray(actions_env_2)

#% ------------------ Define sensor ------------------
sensor_list = []

sensor_list += [  ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=3) ]
sensor = SensorChain(sensor_list)

x0 = sensor.observe(s0)
x1 = sensor.observe(s1)

#% ------------------ Setup experiment ------------------
n_updates_per_frame = 100
n_frames = args.n_updates // n_updates_per_frame
batch_size = args.batch_size

coefs = {'L_genik': args.L_genik}

discrete_cfg = {'groups':args.groups, 'n_embed':args.n_embed}

fnet = GenIKNet(n_actions=4,
                input_shape=x0.shape[1:],
                n_latent_dims=args.latent_dims,
                n_hidden_layers=1,
                n_units_per_layer=32,
                lr=args.learning_rate,
                use_vq=args.use_vq,
                discrete_cfg=discrete_cfg,
                coefs=coefs)

if args.use_two_mazes : 
    x0_env_2 = sensor.observe(s0_env_2)
    x1_env_2 = sensor.observe(s1_env_2)

    ## Concatenate observations from the two mazes
    x0 = np.concatenate([x0, x0_env_2], axis=2)
    x1 = np.concatenate([x1, x1_env_2], axis=2)


#% ------------------ Test Samples ------------------
n_test_samples = args.n_test_samples
test_s0 = s0[-n_test_samples:, :]
test_s1 = s1[-n_test_samples:, :]

test_x0 = sensor.observe(test_s0)
test_x1 = sensor.observe(test_s1)

test_x0 = torch.as_tensor(test_x0[-n_test_samples:, :]).float()
test_x1 = torch.as_tensor(test_x1[-n_test_samples:, :]).float()


test_actions = torch.as_tensor(actions[-n_test_samples:]).float()


def get_batch(x0, x1, a, s0, s1, batch_size=batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx]).float()
    ta = torch.as_tensor(a[idx]).long()
    sx0 = torch.as_tensor(s0[idx]).float()

    if args.L_genik > 0.0 :
        # sample future state k randomly : x_l, a_l, x_t - where x_t is picked randomly k steps ahead
        idx_k = np.random.choice(len(a), batch_size, replace=False)
        tx1 = torch.as_tensor(x1[idx_k]).float()    
        sx1 = torch.as_tensor(s1[idx_k]).float()
        tik = torch.as_tensor(idx_k).long()

    else: #default sampling from buffer
        tx1 = torch.as_tensor(x1[idx]).float()
        sx1 = torch.as_tensor(s1[idx]).float()

    ti = torch.as_tensor(idx).long()   

    return tx0, tx1, ta,  sx0, sx1

get_next_batch = (lambda: get_batch(   x0[:n_samples // 2, :],   x1[:n_samples // 2, :],    a[:n_samples // 2],   s0[:n_samples],  s1[:n_samples]   )  )


type1_evaluations = []
type2_evaluations = []
abstraction_accuracy =[]
abstraction_error = []

test_s0_positions, test_s1_positions, statetogrid = states_pos_to_int(test_s0, test_s1, args.rows, args.cols)

test_s0_positions = torch.as_tensor(test_s0_positions).float()
test_s1_positions = torch.as_tensor(test_s1_positions).float()

#% ------------------ Run Experiment -----------------''
data = []
ncodes = args.n_embed

for frame_idx in tqdm(range(n_frames + 1)):

    print ("Epoch", frame_idx)

    state_transition = torch.zeros(ncodes,4,ncodes)
    tr_lst = []
    trn_lst = []

    for j in range(0,ncodes):

        print ("Code", j)

        tr_lst.append([])
        trn_lst.append([])

    for _ in range(n_updates_per_frame):
        tx0, tx1, ta, ts0, ts1 = get_next_batch()
        fnet.use_vq = args.use_vq
        fnet.train_batch(tx0, tx1, ta)

    with torch.no_grad():
        fnet.eval()

    z0 = fnet.phi(test_x0)
    z1 = fnet.phi(test_x1)

<<<<<<< HEAD
        print('test_s0 shape', test_s0.shape)

        if fnet.use_vq:
            z0, zq_loss0, z_discrete0, ind_0 = fnet.vq_layer(z0)
            z1, zq_loss1, z_discrete1, ind_1 = fnet.vq_layer(z1)

            print('z0 shape', z0.shape)

            zq_loss = zq_loss0 + zq_loss1
            zq_loss = zq_loss.numpy().tolist()
=======
    z0, zq_loss0, z_discrete0, ind_last = fnet.vq_layer(z0)
    z1, zq_loss1, z_discrete1, ind_new = fnet.vq_layer(z1)

    type1_err, type2_err, abs_acc, abs_err = get_eval_error(ind_last, ind_new, test_s0_positions, test_s1_positions)
>>>>>>> 18cff2aec28827e4802d964146bd2035a440e497

    type1_evaluations.append(type1_err)
    type2_evaluations.append(type2_err)
    abstraction_accuracy.append(abs_acc)
    abstraction_error.append(abs_err)

    if args.use_logger:
        logger.record_type1_errors(type1_evaluations)
        logger.record_type2_errors(type2_evaluations)
        logger.record_abstraction_accuracy(abstraction_accuracy)
        logger.record_abstraction_error(abstraction_error)
        logger.save()


    ind_last = ind_last.flatten()
    ind_new = ind_new.flatten()
    # test_actions = test_actions.flatten()

<<<<<<< HEAD
            import ipdb; ipdb.set_trace()
            # # looping through each of the indices
            for j in range(0, ind_last_0.max().item() + 1) :

                state_for_code_0 = test_s0_positions[ind_last_0 == j]
                code_to_state_0.append(state_for_code_0)
            #     import ipdb; ipdb.set_trace()
            #     state_for_code_0 = test_s0_positions[ind_last_0 == j]
            #     code_to_state_0.append(state_for_code_0)

                if args.use_logger:
                    plot_code_to_state_visualization(code_to_state_0, codes_numbers, logger.save_folder, args.n_embed, statetogrid, j, gridsize)
=======
    for j in range(0, ind_last.shape[0]):
        state_transition[ind_last.long()[j], test_actions.long()[j], ind_new.long()[j]] += 1

    for j in range(0, ncodes):
        tr_lst[j] += test_s0_positions[ind_last==j]
        trn_lst[j] += test_s1_positions[ind_new==j]

        if len(tr_lst[j]) > 0:
            print('last', j, tr_lst[j], 'mode', statistics.mode(tr_lst[j]))
        if len(trn_lst[j]) > 0:
            print('next', j, trn_lst[j], 'mode', statistics.mode(trn_lst[j]))
>>>>>>> 18cff2aec28827e4802d964146bd2035a440e497


    ind_last = ind_last.flatten()
    ind_new = ind_new.flatten()

    if frame_idx % 2 == 0:
        print('s last', test_s0_positions)
        print('last', ind_last)
        print('a1', actions)
        print('next', ind_new)
        print('s next', test_s1_positions)  

        # mode_lst = []
        # moden_lst = []
        # for j in range(0,ncodes):
        #     if len(tr_lst[j]) == 0:
        #         mode_lst.append(-1)
        #     else:
        #         mode_lst.append(statistics.mode(tr_lst[j]))
            
        #     if len(trn_lst[j]) == 0:
        #         moden_lst.append(-1)
        #     else:
        #         moden_lst.append(statistics.mode(trn_lst[j]))#torch.Tensor(trn_lst[j]).mode()[0])


        #     print('state transition matrix!')
        #     for a in range(0,3):
        #         for k in range(0,state_transition.shape[0]-1):

        #             if state_transition[k,a].sum().item() > 0:
        #                 print(mode_lst[k], a-1, 'argmax', moden_lst[state_transition[k,a].argmax()], 'num', state_transition[k,a].sum().item())














