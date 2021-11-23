
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


parser = get_parser()

parser.add_argument('--type', type=str, default='genIK', choices=['markov', 'autoencoder', 'genIK'],
                    help='Which type of representation learning method')

parser.add_argument('-n','--n_updates', type=int, default=3000,
                    help='Number of training updates')

parser.add_argument('-r','--rows', type=int, default=6,
                    help='Number of gridworld rows')

parser.add_argument('-c','--cols', type=int, default=6,
                    help='Number of gridworld columns')

parser.add_argument('-w', '--walls', type=str, default='empty', choices=['empty', 'maze', 'spiral', 'loop'],
                    help='The wall configuration mode of gridworld')

parser.add_argument('-l','--latent_dims', type=int, default=128,
                    help='Number of latent dimensions to use for representation')

parser.add_argument('--L_inv', type=float, default=0.0,
                    help='Coefficient for inverse-model-matching loss')

parser.add_argument('--L_genik', type=float, default=0.0,
                    help='Coefficient multi-step inverse dynamics loss')

parser.add_argument('--L_coinv', type=float, default=0.0,
                    help='Coefficient for *contrastive* inverse-model-matching loss')

parser.add_argument('--L_rat', type=float, default=0.0,
                    help='Coefficient for ratio-matching loss')

parser.add_argument('--L_dis', type=float, default=0.0,
                    help='Coefficient for planning-distance loss')

parser.add_argument('--L_ora', type=float, default=0.0,
                    help='Coefficient for oracle distance loss')

parser.add_argument('-lr','--learning_rate', type=float, default=0.003,
                    help='Learning rate for Adam optimizer')

parser.add_argument('--batch_size', type=int, default=100,
                    help='Mini batch size for training updates')

parser.add_argument('--n_test_samples', type=int, default=4000,
                    help='Number of test samples to use for the representation')

parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')

parser.add_argument('-t','--tag', type=str, required=True,
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
parser.add_argument("--type_obs", type=str, default='heatmap', help='image | image_exo_noise | heatmap | heatmap_exo_noise')

parser.add_argument("--use_logger", action="store_true", default=False, help='whether to use logging or not')

parser.add_argument("--exo_noise", action="store_true", default=False, help='whether to use exo noise or not')

parser.add_argument("--use_rgb", action="store_true", default=False, help='whether to use rgb observations')

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


#% ------------------ Generate experiences ------------------

# n_samples = args.n_test_samples
n_samples = 30000
start_state = env.get_state()
states = [start_state]
actions = []

if args.use_two_mazes:
    start_state_env_2 = env_2.get_state()
    states_env_2 = [start_state_env_2]
    actions_env_2 = []
current_state = env.get_state()


if 'image' in args.type_obs:
    obses = [env.get_obs(with_exo_noise=('noise' in args.type_obs))]
    obses_env_2 = [env_2.get_obs(with_exo_noise=('noise' in args.type_obs))]

config = {
          "num_circles": 8,
          "circle_width": 6,
          "circle_motion": 0.05
}

env.set_exo_noise_config(config) 



policy = StationaryStochasticPolicy(len(env.actions), obs_dim=current_state.shape[0])
# two env rollouts - one with policy \pi and another with random action
for step in range(n_samples):
    a = np.random.choice(env.actions)
    next_state, reward, _ = env.step(a)
    current_state = next_state
    states.append(next_state)
    actions.append(a)
    if 'image' in args.type_obs:
        obses.append(env.get_obs(with_exo_noise=('noise' in args.type_obs)))

    if args.use_two_mazes:
        ## Control Maze 2 with Policy \pi
        a2 = policy.sample_action(current_state)    ##### a = policy.get_argmax_action(current_state)
        next_state2, reward2, _ = env_2.step(a2)
        states_env_2.append(next_state2)
        actions_env_2.append(a2)
        if 'image' in args.type_obs:
            obses_env_2.append(env_2.get_obs(with_exo_noise=('noise' in args.type_obs)))

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

sensor_list += [   ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=3) ]
sensor = SensorChain(sensor_list)

if 'heatmap' in args.type_obs:
    x0 = sensor.observe(s0)
    x1 = sensor.observe(s1)

    if args.use_two_mazes : 
        x0_env_2 = sensor.observe(s0_env_2)
        x1_env_2 = sensor.observe(s1_env_2)

elif 'image' in args.type_obs:
    obses = np.stack(obses)
    x0 = obses[:-1]
    x1 = obses[1:]



#% --------------------------------------------------------
if args.use_rgb : 
    ob0 = sensor.observe(s0)
    ob1 = sensor.observe(s1)

    if args.use_two_mazes : 
        ob0_env_2 = sensor.observe(s0_env_2)
        ob1_env_2 = sensor.observe(s1_env_2)
    im0 = env.get_image(ob0, exo_noise=args.exo_noise, corr_noise=args.corr_noise)
    im1 = env.get_image(ob1, exo_noise=args.exo_noise, corr_noise=args.corr_noise)

    x0 = im0
    x1 = im1


    if args.use_two_mazes:
        im0_env_2 = env.get_image(ob0_env_2, exo_noise=args.exo_noise, corr_noise=args.corr_noise)
        im1_env_2 = env.get_image(ob1_env_2, exo_noise=args.exo_noise, corr_noise=args.corr_noise)
        x0_env_2 = im0_env_2
        x1_env_2 = im1_env_2

    ## Concatenate observations from the two mazes
    x0 = np.concatenate([x0, x0_env_2], axis=1)
    x1 = np.concatenate([x1, x1_env_2], axis=1)


#% ------------------ Setup experiment ------------------
n_updates_per_frame = 100
n_frames = args.n_updates // n_updates_per_frame
batch_size = args.batch_size

coefs = {
    'L_inv': args.L_inv,
    'L_coinv': args.L_coinv,
    'L_rat': args.L_rat,
    'L_dis': args.L_dis,
    'L_ora': args.L_ora,
}




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


fnet.print_summary()

n_test_samples = 1000
test_s0 = s0[-n_test_samples:, :]
test_s1 = s1[-n_test_samples:, :]

test_obs0 = sensor.observe(test_s0)
test_obs1 = sensor.observe(test_s1)

if args.use_rgb : 
    test_x0 = test_obs0
    test_x1 = test_obs1
    test_x0 = torch.as_tensor(test_x0).float()
    test_x1 = torch.as_tensor(test_x1).float()

else : 
    test_x0 = torch.as_tensor(x0[-n_test_samples:, :]).float()
    test_x1 = torch.as_tensor(x1[-n_test_samples:, :]).float()


test_a = torch.as_tensor(a[-n_test_samples:]).long()

env.reset_agent()
state = env.get_state()
obs = sensor.observe(state)



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
gridsize = args.rows * args.cols
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

test_s0_positions, test_s1_positions, statetogrid = states_pos_to_int(test_s0, test_s1, args.rows, args.cols)

def test_rep(fnet, step,  test_s0, test_s1, test_s0_positions, test_s1_positions, frame_idx, n_frames):
    with torch.no_grad():
        fnet.eval()
        z0 = fnet.phi(test_x0)
        z1 = fnet.phi(test_x1)

        if fnet.use_vq:
            z0, zq_loss0, z_discrete0, ind_0 = fnet.vq_layer(z0)
            z1, zq_loss1, z_discrete1, ind_1 = fnet.vq_layer(z1)

            zq_loss = zq_loss0 + zq_loss1
            zq_loss = zq_loss.numpy().tolist()

            type1_err, type2_err, abs_acc, abs_err = get_eval_error(ind_0, ind_1, test_s0_positions, test_s1_positions)

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
        else:
            zq_loss = 0.

            loss_info = {
                'step': step,
                'L_inv': fnet.inverse_loss(z0, z1, test_a).numpy().tolist(),
                'L_coinv': fnet.contrastive_inverse_loss(z0, z1, test_a).numpy().tolist(),
                'L_vq': zq_loss,#fnet.compute_entropy_loss(z0, z1, test_a).numpy().tolist(),
                # 'L': (fnet.compute_loss(z0, z1, test_a, torch.zeros((2 * len(z0)))) + zq_loss).numpy().tolist(),
                'type1_error': type1_err,
                'type2_error': type2_err
            }


    return  step, type1_err, type2_err, z_discrete0, z_discrete1




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



#% ------------------ Run Experiment -----------------''
data = []
for frame_idx in tqdm(range(n_frames + 1)):
    for _ in range(n_updates_per_frame):

        tx0, tx1, ta, ts0, ts1 = get_next_batch()
        fnet.use_vq = args.use_vq
        fnet.train_batch(tx0, tx1, ta)

    step, type1_err, type2_err, z_discrete0, z_discrete1 = test_rep(fnet, frame_idx * n_updates_per_frame,  test_s0, test_s1, test_s0_positions, test_s1_positions, frame_idx, n_frames)

