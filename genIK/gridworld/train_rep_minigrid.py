import imageio
import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import seeding
import sys
import torch
from tqdm import tqdm
# import wandb
# import plotly.express as px
from models.featurenet import FeatureNet
from models.autoencoder import AutoEncoder
from models.geniknet import GenIKNet
from repvis import RepVisualization, CleanVisualization #, DiscreteRepVisualization
from visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld, MazeWorld, SpiralWorld, LoopWorld
from visgrid.utils import get_parser, MI
from visgrid.sensors import *
from visgrid.gridworld.distance_oracle import DistanceOracle
from det_tabular_mdp_builder import DetTabularMDPBuilder
from value_iteration import ValueIteration
from utils import Logger, plot_state_visitation, plot_code_to_state_visualization
import random
from matplotlib import pyplot as plt

from minigrid.gridworld1 import GridWorld1
from minigrid.gridworld2 import GridWorld2
from minigrid.gridworld_wrapper import GridWorldWrapper

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

parser.add_argument('--batch_size', type=int, default=2048,
                    help='Mini batch size for training updates')

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

parser.add_argument('--groups', type=int, default=1,
                    help='No. of groups to use for VQ-VAE')

parser.add_argument('--n_embed', type=int, default=10,
                    help='No. of embeddings')
# Clustering layer
parser.add_argument('--use_proto', action='store_true',
                    help='Use prototypes-based discritization after the phi network')

parser.add_argument("--type_obs", type=str, default='image', help='image | image_exo_noise | heatmap | heatmap_exo_noise')

parser.add_argument("--use_logger", action="store_true", default=False, help='whether to use logging or not')

parser.add_argument("--m_step_mode", type=str, default='random_future', choices = [ 'random_future', 'indep_future_states'  ], help='sampling of future k states, using multi-step-inv-kinematics mode')

parser.add_argument('--corr_noise', type=float, default=0.0, help='value corresponds to correlation length of noise; default 0.0 means i.i.d noise')
parser.add_argument("--exo_noise", action="store_true", default=False, help='whether to use exo noise or not')
parser.add_argument("--use_rgb", action="store_true", default=False, help='whether to use rgb observations')

parser.add_argument("--folder", type=str, default='./results/')


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

seeding.seed(args.seed)


#### Make minigrid env
with open("minigrid/config1.json") as f:
    env_config = json.load(f)

minigrid_env = GridWorld1(env_config)
env = GridWorldWrapper(minigrid_env, env_config)
env_name = 'minigrid1'

#% ------------------ LOGGERS ------------------
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


log_dir = 'results/logs/' + str(args.tag)
vid_dir = 'results/videos/' + str(args.tag)
maze_dir = 'results/mazes/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)
if args.video:
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(maze_dir, exist_ok=True)
    # video_filename = vid_dir + '/video-{}.mp4'.format(args.seed)
    # image_filename = vid_dir + '/final-{}.png'.format(args.seed)
    maze_file = maze_dir + '/maze-{}.png'.format(args.seed)

    if args.use_logger : 
        video_filename = logger.save_folder + '/video-{}.mp4'.format(args.seed)
        image_filename = logger.save_folder + '/final-{}.png'.format(args.seed)
log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
    arg_file.write(repr(args))
# wandb.config.update(vars(args))



# cmap = 'Set3'
cmap = None

#% ------------------ Build Deterministic Tabular MDP Model ------------------
horizon = 20000
model = DetTabularMDPBuilder(actions=env.actions, horizon=horizon, gamma=1.0)  

#% ------------------ Generate experiences ------------------
n_samples = 20000


start_obs, info = env.reset()
states = [info['endogenous_state']]
actions = []


obses = [start_obs]

# config = {
#           "num_circles": 8,
#           "circle_width": 6,
#           "circle_motion": 0.05
# }
# env.set_exo_noise_config(config)
# im = env.plot()
# plt.imshow(im)
# plt.show()

for step in range(horizon):
    a = np.random.choice(env.actions)
    # next_state, reward, _ = env.step(a)
    obs, reward, done, info = env.step(a)

    states.append(info['endogenous_state'])
    actions.append(a)
    obses.append(obs)

    # model.add_state(state=tuple(next_state) , timestep=step)
    # model.add_transition(tuple(current_state), a, tuple(next_state))
    # model.add_reward(tuple(current_state-1), a, float(reward))
    # current_state = next_state
# model.finalize()
# value_it = ValueIteration()
# q_val = value_it.do_value_iteration(tabular_mdp=model, min_reward_val=0.0)
# expected_ret = q_val[(0, (0, 0))].max()

import ipdb; ipdb.set_trace()

states = np.stack(states)
s0 = np.asarray(states[:-1, :])
s1 = np.asarray(states[1:, :])
a = np.asarray(actions)

MI_max = MI(s0, s0)

# ax = env.plot()
# xx = s0[:, 1] + 0.5
# yy = s0[:, 0] + 0.5
# ax.scatter(xx, yy, c=c0)

# Confirm that we're covering the state space relatively evenly
# if args.use_logger:
#     plot_state_visitation(states[:,0], states[:,1], logger.save_folder, bins=6)

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

if args.type == 'genIK':
    discrete_cfg = {'groups':args.groups, 'n_embed':args.n_embed}
    fnet = GenIKNet(n_actions=4,
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


n_test_samples = 2
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

    if args.use_vq:
        repvis = DiscreteRepVisualization(env,
                          obs,
                          batch_size=n_test_samples,
                          n_dims=2,
                          colors=test_c,
                          cmap=cmap)

    elif not args.cleanvis:
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
        if args.type =='genIK':
            z0 = fnet.phi(test_x0)
            z1 = fnet.phi(test_x1)

            if fnet.use_vq:
                z0, zq_loss0, z_discrete0, ind_0 = fnet.vq_layer(z0)
                z1, zq_loss1, z_discrete1, ind_1 = fnet.vq_layer(z1)

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
    return [r.numpy() for r in results] + [text], step, type1_err, type2_err, z_discrete0, z_discrete1




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
    if args.use_vq:
        frame = repvis.update_plots(*test_results, z_discrete0, z_discrete1, env, args.exo_noise)
    else:
        frame = repvis.update_plots(*test_results)
    # wandb.log({'original repo viz': wandb.Image(frame)})
    data.append(frame)

if args.video:
    if args.use_logger:
        imageio.mimwrite(video_filename, data, fps=15)
        # wandb.log({"video": wandb.Video(video_filename)})
        imageio.imwrite(image_filename, data[-1])



log.close()


# wandb_dir = wandb.run.dir[:-len('/files')]
# wandb.run.finish()

# # sync offline run to wandb
# if os.environ["WANDB_MODE"] == "offline":
#     os.system("wandb sync "+wandb_dir)
