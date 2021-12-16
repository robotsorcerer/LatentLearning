import imageio
import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import seeding
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
# import wandb
# import plotly.express as px
from models.featurenet import FeatureNet
from models.autoencoder import AutoEncoder
from models.repnet import RepNet
from repvis import RepVisualization, CleanVisualization
from visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld, MazeWorld, SpiralWorld, LoopWorld
from visgrid.utils import get_parser, MI
from visgrid.sensors import *
from visgrid.gridworld.distance_oracle import DistanceOracle
from det_tabular_mdp_builder import DetTabularMDPBuilder
from value_iteration import ValueIteration
import pandas as pd
import time
from collections import deque
from datetime import datetime
from models.nnutils import Reshape
from models.nullabstraction import NullAbstraction
from models.phinet import PhiNet
from models.vq_layer_kmeans import Quantize
from agents.randomagent import RandomAgent
from agents.dqnagent import DQNAgent
from tensorboardX import SummaryWriter
from utils import Logger, plot_state_visitation, plot_code_to_state_visualization
import random

parser = get_parser()

parser.add_argument('--type', type=str, default='repnet', choices=['markov', 'autoencoder', 'repnet'],
                    help='Which type of representation learning method')

parser.add_argument('-n','--n_updates', type=int, default=3000,
                    help='Number of training updates')

parser.add_argument('-H','--horizon', type=int, default=20000,
                    help='Number of samples to collect based on horizon')

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


parser.add_argument('--n_test_samples', type=int, default=30000,
                    help='Mini batch size for training updates for the representation')

parser.add_argument('--batch_size', type=int, default=2048,
                    help='Mini batch size for training updates for the representation')

parser.add_argument('--batch_size_agent_train', type=int, default=32,
                    help='Mini batch size for training updates for the agent')

parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')

parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')

parser.add_argument('-v','--video', action='store_true',
                    help="Save training video")

parser.add_argument('--no_graphics', action='store_true',
                    help='Turn off graphics (e.g. for running on cluster)')

parser.add_argument('--save', action='store_true',
                    help='Save final network weights')

parser.add_argument('--cleanvis', action='store_true',
                    help='Switch to representation-only visualization')

parser.add_argument('--no_sigma', action='store_true',
                    help='Turn off sensors and just use true state; i.e. x=s')

parser.add_argument('--rearrange_xy', action='store_true',
                    help='Rearrange discrete x-y positions to break smoothness')
# VQ Discrete Layer
parser.add_argument('--use_vq', action='store_true',
                    help='Use VQ layer after the phi network')

parser.add_argument('--groups', type=int, default=2,
                    help='No. of groups to use for VQ-VAE')

parser.add_argument('--n_embed', type=int, default=50,
                    help='No. of embeddings')
# Clustering layer
parser.add_argument('--use_proto', action='store_true',
                    help='Use prototypes-based discritization after the phi network')

parser.add_argument("--use_logger", action="store_true", default=False, help='whether to use logging or not')

parser.add_argument("--m_step_mode", type=str, default='random_future', choices = [ 'random_future', 'indep_future_states'  ], help='sampling of future k states, using multi-step-inv-kinematics mode')

parser.add_argument("--exo_noise", action="store_true", default=False, help='whether to use exo noise or not')

parser.add_argument("--use_rgb", action="store_true", default=False, help='whether to use rgb observations')

parser.add_argument("--folder", type=str, default='./results/')

#additional args for training the agent
parser.add_argument('--xy_noise', action='store_true', help='Add truncated gaussian noise to x-y positions')

parser.add_argument('--no_phi', action='store_true', help='Turn off abstraction and just use observed state; i.e. ϕ(x)=x')

parser.add_argument('-a','--agent', type=str, default='dqn', choices=['random','dqn'], help='Type of agent to train')

parser.add_argument("--vis_features", action="store_true", default=False, help='To visualize the learnt features for the agent during training')

parser.add_argument('-trials','--n_trials', type=int, default=5, help='Number of trials')

parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of steps per episode')

parser.add_argument('--train_phi', action='store_true', help='Allow simultaneous training of abstraction')

parser.add_argument('--n_episodes', type=int, default=300, help='Number of episodes per trial')

parser.add_argument('--gamma', type=float, default=0.99, help='Discount Factor')

parser.add_argument("--use_goal_conditioned", action="store_true", default=False, help='goal conditioning with the discrete latents')

parser.add_argument("--type_obs", type=str, default='heatmap', help='image | image_exo_noise | heatmap | heatmap_exo_noise')

# yapf: enable
if 'ipykernel' in sys.argv[0]:
    arglist = [
        '--spiral', '--tag', 'test-spiral', '-r', '6', '-c', '6', '--L_ora', '1.0', '--video'
    ]
    args = parser.parse_args(arglist)
else:
    args = parser.parse_args()

# use only on discretization
assert args.use_proto + args.use_vq < 2

if args.no_graphics:
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
# wandb.init(project='GridWorld', entity="markov-discrete", name=args.tag)
log_dir = 'results/logs/' + str(args.tag)
vid_dir = 'results/videos/' + str(args.tag)
maze_dir = 'results/mazes/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)

if args.video:
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(maze_dir, exist_ok=True)
    video_filename = vid_dir + '/video-{}.mp4'.format(args.seed)
    image_filename = vid_dir + '/final-{}.png'.format(args.seed)
    maze_file = maze_dir + '/maze-{}.png'.format(args.seed)

log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
    arg_file.write(repr(args))
# wandb.config.update(vars(args))

seeding.seed(args.seed)

#% ------------------ Define MDP ------------------
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
    env = GridWorld(rows=args.rows, cols=args.cols)
    env_name = 'gridworld'
    # env = RingWorld(2,4)
    # env = TestWorld()
    # env.add_random_walls(10)
if args.use_logger:
    file_name = "%s_%s_%s" % (args.type, env_name, str(args.seed))
    if args.exo_noise :
        exogenous = 'exo_noise'
    else:
        exogenous = 'no_exo_noise'
    if args.use_rgb:
        obs_type = 'rgb_obs'
    else: 
        obs_type = 'obs_map'
    logger = Logger(args, experiment_name=args.tag, environment_name=env_name, type_decoder=args.type,  obs_type = obs_type, use_exo = exogenous, groups = 'groups_' + str(args.groups) + '_embed_' + str(args.n_embed), folder=args.folder)
    logger.save_args(args)
    print('Saving to', logger.save_folder)
else:
    logger = None

# cmap = 'Set3'
cmap = None
#% ------------------ Build Deterministic Tabular MDP Model ------------------
horizon=args.horizon
model = DetTabularMDPBuilder(actions=env.actions, horizon=horizon, gamma=1.0)  
#% ------------------ Generate experiences ------------------
n_samples = 30000
start_state = env.get_state()
states = [start_state]
actions = []

current_state = env.get_state()
model.add_state(state=tuple((0,0)), timestep=0)

for step in range(horizon):
    a = np.random.choice(env.actions)
    next_state, reward, _ = env.step(a)

    states.append(next_state)
    actions.append(a)

    if 'image' in args.type_obs:
        obses.append(env.get_obs(with_exo_noise=('noise' in args.type_obs), change_noise=False))
    # model.add_state(state=tuple(next_state) , timestep=step)
    # model.add_transition(tuple(current_state), a, tuple(next_state))
    # model.add_reward(tuple(current_state-1), a, float(reward))
    # current_state = next_state
# model.finalize()
# value_it = ValueIteration()
# q_val = value_it.do_value_iteration(tabular_mdp=model, min_reward_val=0.0)
# expected_ret = q_val[(0, (0, 0))].max()


states = np.stack(states)
s0 = np.asarray(states[:-1, :])
c0 = s0[:, 0] * env._cols + s0[:, 1]
s1 = np.asarray(states[1:, :])
a = np.asarray(actions)

unique_states = set([tuple(_) for _ in states.tolist()])
all_states = np.asarray(list(set(unique_states)))

MI_max = MI(s0, s0)

# ax = env.plot()
# xx = s0[:, 1] + 0.5
# yy = s0[:, 0] + 0.5
# ax.scatter(xx, yy, c=c0)

# Confirm that we're covering the state space relatively evenly
if args.use_logger:
    plot_state_visitation(states[:,0], states[:,1], logger.save_folder, bins=6)



#% ------------------ Define sensor ------------------
sensor_list = []
if args.rearrange_xy:
    sensor_list.append(RearrangeXYPositionsSensor((env._rows, env._cols)))

if not args.no_sigma:
    sensor_list += [
        OffsetSensor(offset=(0.5, 0.5)),
        NoisySensor(sigma=0.05),
        ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=3),
        # ResampleSensor(scale=2.0),
        BlurSensor(sigma=0.6, truncate=1.),
        NoisySensor(sigma=0.01)
    ]
sensor = SensorChain(sensor_list)

if 'heatmap' in args.type_obs:
    x0 = sensor.observe(s0)
    x1 = sensor.observe(s1)
elif 'image' in args.type_obs:
    obses = np.stack(obses)
    x0 = obses[:-1]
    x1 = obses[1:]


if args.use_rgb : 
    ob0 = sensor.observe(s0)
    ob1 = sensor.observe(s1)
    im0 = env.get_image(ob0, exo_noise=args.exo_noise, corr_noise=args.corr_noise)
    im1 = env.get_image(ob1, exo_noise=args.exo_noise, corr_noise=args.corr_noise)

    x0 = im0
    x1 = im1
# else:
#     x0 = ob0
#     x1 = ob1

# for viz
all_obs = sensor.observe(all_states)
all_obs = torch.as_tensor(all_obs).float()

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

if args.type == 'repnet':
    discrete_cfg = {'groups':args.groups, 'n_embed':args.n_embed}
    fnet = RepNet(n_actions=4,
                  input_shape=x0.shape[1:],
                  n_latent_dims=args.latent_dims,
                  n_hidden_layers=1,
                  n_units_per_layer=32,
                  lr=args.learning_rate,
                  use_vq=args.use_vq,
                  use_proto=args.use_proto,
                  discrete_cfg=discrete_cfg,
                  coefs=coefs)

elif args.type == 'autoencoder':
    fnet = AutoEncoder(n_actions=4,
                       input_shape=x0.shape[1:],
                       n_latent_dims=args.latent_dims,
                       n_hidden_layers=1,
                       n_units_per_layer=32,
                       lr=args.learning_rate,
                       coefs=coefs)


fnet.print_summary()

n_test_samples = args.n_test_samples
test_s0 = s0[-n_test_samples:, :]
test_s1 = s1[-n_test_samples:, :]

test_obs0 = sensor.observe(test_s0)
test_obs1 = sensor.observe(test_s1)

if args.use_rgb : 
    ### testing here : TODO
    test_x0 = test_obs0
    test_x1 = test_obs1
    # test_x0 = env.get_image(test_obs0, exo_noise=args.exo_noise, corr_noise=args.corr_noise)
    # test_x1 = env.get_image(test_obs1, exo_noise=args.exo_noise, corr_noise=args.corr_noise)
    test_x0 = torch.as_tensor(test_x0).float()
    test_x1 = torch.as_tensor(test_x1).float()

else : 
    test_x0 = torch.as_tensor(x0[-n_test_samples:, :]).float()
    test_x1 = torch.as_tensor(x1[-n_test_samples:, :]).float()


test_a = torch.as_tensor(a[-n_test_samples:]).long()
test_i = torch.arange(n_test_samples).long()
test_c = c0[-n_test_samples:]

oracle = DistanceOracle(env)

env.reset_agent()
state = env.get_state()
obs = sensor.observe(state)

if args.video:

    if not args.cleanvis:
        repvis = RepVisualization(env,
                                  obs,
                                  batch_size=n_test_samples,
                                  n_dims=2,
                                  colors=test_c,
                                  cmap=cmap)
    else:
        repvis = CleanVisualization(env,
                                    obs,
                                    batch_size=n_test_samples,
                                    n_dims=2,
                                    colors=test_c,
                                    cmap=cmap)

def get_batch(x0, x1, a, s0, s1, m_step_sampling=args.m_step_mode, batch_size=batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx]).float()
    ta = torch.as_tensor(a[idx]).long()
    sx0 = torch.as_tensor(s0[idx]).float()

    if args.m_step_mode == 'indep_future_states':
        # sample future state k randomly : x_l, a_l, x_t - where x_t is picked randomly k steps ahead, but != x_l
        buffer_range = np.arange(len(a))
        k_x0 = np.setdiff1d(buffer_range, idx)
        idx_k = np.random.choice(len(k_x0), batch_size, replace=False)
        tx1 = torch.as_tensor(x1[idx_k]).float()
        sx1 = torch.as_tensor(s1[idx_k]).float()
        tik = torch.as_tensor(idx_k).long()

    elif args.m_step_mode =='random_future':
        # sample future state k randomly : x_l, a_l, x_t - where x_t is picked randomly k steps ahead
        idx_k = np.random.choice(len(a), batch_size, replace=False)
        tx1 = torch.as_tensor(x1[idx_k]).float()    
        sx1 = torch.as_tensor(s1[idx_k]).float()
        tik = torch.as_tensor(idx_k).long()

    else: #default sampling from buffer
        tx1 = torch.as_tensor(x1[idx]).float()
        sx1 = torch.as_tensor(s1[idx]).float()

    ti = torch.as_tensor(idx).long()    
    return tx0, tx1, ta, idx, sx0, sx1


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
codes_numbers = random.sample(range(0, args.n_embed - 1),  5)

def test_rep(fnet, step,  test_s0, test_s1, test_s0_positions, test_s1_positions, frame_idx, n_frames):
    with torch.no_grad():
        fnet.eval()
        if args.type =='repnet':
            z0 = fnet.phi(test_x0)
            z1 = fnet.phi(test_x1)

            if fnet.use_vq:
                z0, zq_loss0, ind_0 = fnet.vq_layer(z0)
                z1, zq_loss1, ind_1 = fnet.vq_layer(z1)

                ind_last_0 = ind_0.flatten()
                ind_last_1 = ind_1.flatten()

                code_to_state_0 = []
                code_to_state_1 = []

                # looping through each of the indices
                for j in range(0, ind_last_0.max().item() + 1) :
                    state_for_code_0 = test_s0_positions[ind_last_0 == j]
                    code_to_state_0.append(state_for_code_0)

                    state_for_code_1 = test_s0_positions[ind_last_1 == j]
                    code_to_state_1.append(state_for_code_1)

                    if args.use_logger:
                        plot_code_to_state_visualization(code_to_state_0, codes_numbers, logger.save_folder, args.n_embed, statetogrid, j, gridsize)

                zq_loss = zq_loss0 + zq_loss1
                zq_loss = zq_loss.numpy().tolist()

                type1_err, type2_err, abs_acc, abs_err = get_eval_error(ind_0, ind_1, test_s0_positions, test_s1_positions)
                # type1_err, type2_err = get_eval_error(ind_0, ind_1, test_s0, test_s1)

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

            elif fnet.use_proto:
                z0, zq_loss0, _ = fnet.vq_layer(z0)
                z1, zq_loss1, _ = fnet.vq_layer(z1)
                zq_loss = zq_loss0 + zq_loss1
                zq_loss = zq_loss.numpy().tolist()
            else:
                zq_loss = 0.
            # yapf: disable
            loss_info = {
                'step': step,
                'L_inv': fnet.inverse_loss(z0, z1, test_a).numpy().tolist(),
                'L_coinv': fnet.contrastive_inverse_loss(z0, z1, test_a).numpy().tolist(),
                'L_vq': zq_loss,#fnet.compute_entropy_loss(z0, z1, test_a).numpy().tolist(),
                'L': (fnet.compute_loss(z0, z1, test_a, torch.zeros((2 * len(z0)))) + zq_loss).numpy().tolist(),
                'MI': MI(test_s0, z0.numpy()) / MI_max, 
                'type1_error': type1_err,
                'type2_error': type2_err
            }
            # wandb.log(loss_info)
            # yapf: enable

        elif args.type == 'autoencoder':
            z0 = fnet.encode(test_x0)
            z1 = fnet.encode(test_x1)

            loss_info = {
                'step': step,
                'L': fnet.compute_loss(test_x0).numpy().tolist(),
            }
            ### wandb.log(loss_info)

    json_str = json.dumps(loss_info)
    log.write(json_str + '\n')
    log.flush()

    text = '\n'.join([key + ' = ' + str(val) for key, val in loss_info.items()])

    results = [z0, z1, z1, test_a, test_a]
    return [r.numpy() for r in results] + [text], step, type1_err, type2_err, ind_0, ind_1




def get_eval_error (z0, z1, s0, s1):
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
        # Z_comp = Z[0] * Z[1] * Z[2] * Z[3]
        Z_comp = torch.prod(Z)

        S = s0[i] == s1[i]
        S = S.long()
        # S_comp = S[0] * S[1]
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

#% ------------------ Run Experiment ------------------

data = []
for frame_idx in tqdm(range(n_frames + 1)):
    for _ in range(n_updates_per_frame):

        tx0, tx1, ta, idx, ts0, ts1 = get_next_batch()

        tdist = torch.cat([
            torch.as_tensor(oracle.pairwise_distances(idx, s0, s1)).squeeze().float(),
            torch.as_tensor(oracle.pairwise_distances(idx, s0, np.flip(s1))).squeeze().float()
        ], dim=0) # yapf: disable
        # h = np.histogram(tdist, bins=36)[0]
        ###### warm up without quantization for 10 epochs
        # if frame_idx < 10:
        #     fnet.use_vq = False
        # else:
        #     fnet.use_vq = args.use_vq
        # '''
        # without warm up
        # '''
        fnet.use_vq = args.use_vq
        fnet.use_proto = args.use_proto
        fnet.train_batch(tx0, tx1, ta, idx)

    test_results, step, type1_err, type2_err, z_discrete0, z_discrete1 = test_rep(fnet, frame_idx * n_updates_per_frame,  test_s0, test_s1, test_s0_positions, test_s1_positions, frame_idx, n_frames)

if args.video:
    imageio.mimwrite(video_filename, data, fps=15)
    # wandb.log({"video": wandb.Video(video_filename)})
    imageio.imwrite(image_filename, data[-1])

if args.use_logger:
    fnet.phi.save('phi-{}'.format(args.seed),   logger.save_folder)
    if args.use_vq:
        torch.save(fnet.vq_layer.state_dict(), logger.save_folder +'/phi-vq.pt')
    elif args.use_proto:
        torch.save(fnet.proto.state_dict(), logger.save_folder + '/phi-proto.pt')







#% ------------------ Run Experiment for training the agent ------------------
if args.walls in ['empty', 'spiral', 'loop']:
    maze_dir = 'maze_' + args.walls
else:
    maze_dir = 'maze' + str(args.seed)

log_dir = 'results/scores/{}/pretrain_3k/'.format(maze_dir) + str(args.tag)
os.makedirs(log_dir, exist_ok=True)
log = open(log_dir + '/scores-{}-{}.txt'.format(args.agent, args.seed), 'w')

gamma = args.gamma

if args.xy_noise:
    sensor_list.append(NoisySensor(sigma=0.2, truncation=0.4))


if args.no_phi:
    phinet = NullAbstraction(-1, args.latent_dims)
else:
    x0 = sensor.observe(env.get_state())
    phinet = PhiNet(input_shape=x0.shape,
                    n_latent_dims=args.latent_dims,
                    n_hidden_layers=1,
                    n_units_per_layer=32)

    modelfile = logger.save_folder +'/phi-{}_latest.pytorch'.format(args.seed)
    phinet.load(modelfile)

    if args.use_vq:
        vq_layer = Quantize(args.latent_dims, args.n_embed, args.groups)
        vq_path = logger.save_folder + '/phi-vq.pt'

        vq_layer.load_state_dict(
            torch.load(vq_path)
        )
    else:
        vq_layer = None

    # visualize the features to ensure the representation is rightly loaded
    if args.vis_features:
        n_test_samples = 2000
        states = [env.get_state()]
        actions = []
        for t in range(n_test_samples):
            a = np.random.choice(env.actions)
            s, _, _ = env.step(a)
            states.append(s)
            actions.append(a)
        states = np.stack(states)
        s0 = np.asarray(states[:-1, :])
        c0 = s0[:, 0] * env._cols + s0[:, 1]
        s1 = np.asarray(states[1:, :])
        x1 = sensor.observe(s1)
        test_x1 = torch.as_tensor(x1[-n_test_samples:, :]).float()
        with torch.no_grad():
            phinet.eval()

            z1 = phinet(test_x1)

            if args.use_vq:
                vq_layer.eval()
                z1, zq_loss1, z_ind = vq_layer(z1)
                vq_layer.train()

        test_c = c0[-n_test_samples:]
        repvis = CleanVisualization(env,
                                    x0,
                                    batch_size=n_test_samples,
                                    n_dims=2,
                                    colors=test_c)
        repvis.update_plots(z1, [], [], [], [], [])
        time.sleep(2)
        # after visualization
        phinet.train()
        plt.close()

seeding.seed(args.seed, np)

#%% ------------------ Load agent ------------------
n_actions = 4
if args.agent == 'random':
    agent = RandomAgent(n_actions=n_actions)

elif args.agent == 'dqn':
    agent = DQNAgent(n_features=args.latent_dims,
                     n_actions=n_actions,
                     phi=phinet,
                     use_goal_conditioned=args.use_goal_conditioned,
                     lr=args.learning_rate,
                     batch_size=args.batch_size_agent_train,
                     train_phi=args.train_phi,
                     gamma=gamma,
                     decay_period=args.n_episodes * args.max_steps * 0.6,
                     vq_layer=vq_layer,
                     epsilon=0.1 # set a larger epsilon
                     )  
else:
    assert False, 'Invalid agent type: {}'.format(args.agent)

#%% ------------------ Train agent ------------------
if args.video:
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    fig.show()

    def plot_value_function(ax):
        s = np.asarray([[np.asarray([x, y]) for x in range(args.cols)] for y in range(args.rows)])
        v = np.asarray(agent.v(s).detach().numpy())
        xy = OffsetSensor(offset=(0.5, 0.5)).observe(s).reshape(args.cols, args.rows, -1)
        ax.contourf(np.arange(0.5, args.cols + 0.5),
                    np.arange(0.5, args.rows + 0.5),
                    v,
                    vmin=-10,
                    vmax=0)

    def plot_states(ax):
        data = pd.DataFrame(agent.replay.memory)
        data[['x.r', 'x.c']] = pd.DataFrame(data['x'].tolist(), index=data.index)
        data[['xp.r', 'xp.c']] = pd.DataFrame(data['xp'].tolist(), index=data.index)
        sns.scatterplot(data=data,
                        x='x.c',
                        y='x.r',
                        hue='done',
                        style='done',
                        markers=True,
                        size='done',
                        size_order=[1, 0],
                        ax=ax,
                        alpha=0.3,
                        legend=False)
        ax.invert_yaxis()

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir1 = 'results/tb/walls_{}_vq_{}_'.format(args.walls, args.use_vq) + current_time
tb_logger = SummaryWriter(logdir=log_dir1)

def eval_agent(goal_pos, env, args, sensor, agent, tb_logger, ep_num):
    eval_reward = []
    for i in range(args.rows):
        for j in range(args.cols):
            if i != goal_pos[0] and j != goal_pos[1]:
                agent_pos = [i, j]
                env.reset_agent(pos=agent_pos)
                ep_rewards = []
                for step in range(args.max_steps):
                    s = env.get_state()
                    x = sensor.observe(s)
                    a = agent.greedy_act(x)
                    sp, r, done = env.step(a)
                    ep_rewards.append(r)
                    if done:
                        break
                eval_reward.append(sum(ep_rewards))
    mean_eval_reward = np.mean(eval_reward)
    tb_logger.add_scalar('eval/reward', mean_eval_reward, ep_num)
    eval_array = np.array(eval_reward)
    success = np.sum(eval_array > - args.max_steps) / len(eval_array)
    tb_logger.add_scalar('eval/success', success, ep_num)

    return mean_eval_reward, success


for trial in tqdm(range(args.n_trials), desc='trials'):
    # if args.walls in ['empty']:
    #     goal_pos = None
    # else:
    #     goal_pos = [2, 3]
    goal_pos = [2, 3]
    env.reset_goal(goal_pos=goal_pos)
    agent.reset()
    total_reward = 0
    total_steps = 0
    losses = []
    rewards = []
    value_fn = []
    reward_lst = deque(maxlen=500)

    all_mean_eval_rewards = []
    all_eval_success = []
    for episode in tqdm(range(args.n_episodes), desc='episodes'):
        env.reset_agent()
        ep_rewards = []
        for step in range(args.max_steps):
            s = env.get_state()
            x = sensor.observe(s)
            a = agent.act(x)
            sp, r, done = env.step(a)
            xp = sensor.observe(sp)
            ep_rewards.append(r)
            if args.video:
                value_fn.append(agent.v(x))
            total_reward += r

            loss = agent.train(x, a, r, xp, done)
            losses.append(loss)

            if done:
                break

        if args.video:
            [a.clear() for a in ax]
            plot_value_function(ax[0])
            env.plot(ax[0])
            ax[1].plot(value_fn)
            ax[2].plot(rewards, c='C3')
            ax[3].plot(losses, c='C1')
            # plot_states(ax[3])
            ax[1].set_ylim([-10, 0])
            fig.canvas.draw()
            fig.canvas.flush_events()

        # eval every 100 episodes
        if episode % 10 == 0:
            mean_eval_reward, eval_success = eval_agent(goal_pos, env, args, sensor, agent, tb_logger, episode)
            all_mean_eval_rewards.append(mean_eval_reward)
            all_eval_success.append(eval_success)

        total_steps += step
        score_info = {
            'trial': trial,
            'episode': episode,
            'reward': sum(ep_rewards),
            'total_reward': total_reward,
            'total_steps': total_steps,
            'steps': step
        }


        rewards.append(total_reward)
        reward_lst.append(sum(ep_rewards))
        wd_size = 50
        mean200 = np.mean(list(reward_lst)[-wd_size:])
        success = np.sum(np.array(reward_lst)[-wd_size:] > -args.max_steps)
        success = success / wd_size
        # record in tensorboard
        tb_logger.add_scalar("train/reward", mean200, episode)
        tb_logger.add_scalar("train/success", success, episode)

        if args.use_logger:
            logger.record_mean_rewards(mean200)
            logger.record_returns(rewards)
            logger.record_losses(losses)
            logger.record_eval_return(all_mean_eval_rewards)
            logger.record_eval_success(all_eval_success)

            logger.save_agent_train_metric()

        json_str = json.dumps(score_info)
        log.write(json_str + '\n')
        log.flush()
print('\n\n')


