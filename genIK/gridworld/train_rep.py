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


from repvis import RepVisualization, CleanVisualization
from visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld, MazeWorld, SpiralWorld, LoopWorld
from visgrid.utils import get_parser, MI
from visgrid.sensors import *
from visgrid.gridworld.distance_oracle import DistanceOracle

from det_tabular_mdp_builder import DetTabularMDPBuilder
from value_iteration import ValueIteration


from utils import Logger





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

parser.add_argument('-l','--latent_dims', type=int, default=256,
                    help='Number of latent dimensions to use for representation')

parser.add_argument('--L_inv', type=float, default=1.0,
                    help='Coefficient for inverse-model-matching loss')

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

parser.add_argument('--groups', type=int, default=2,
                    help='No. of groups to use for VQ-VAE')

parser.add_argument('--n_embed', type=int, default=10,
                    help='No. of embeddings')

# Clustering layer
parser.add_argument('--use_proto', action='store_true',
                    help='Use prototypes-based discritization after the phi network')





parser.add_argument("--use_logger", action="store_true", default=False, help='whether to use logging or not')




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

    logger = Logger(args, experiment_name=args.tag, environment_name=env_name, type_decoder=args.type,   groups = 'groups_' + str(args.groups) + '_embed_' + str(args.n_embed), folder="./results/")
    logger.save_args(args)

    print('Saving to', logger.save_folder)
else:
    logger = None



# cmap = 'Set3'
cmap = None


#% ------------------ Build Deterministic Tabular MDP Model ------------------
horizon=20000
model = DetTabularMDPBuilder(actions=env.actions, horizon=horizon, gamma=1.0)  

#% ------------------ Generate experiences ------------------
n_samples = 20000
states = [env.get_state()]
actions = []

current_state = env.get_state()
model.add_state(state=tuple((0,0)), timestep=0)


# for step in range(1, horizon + 1):
for step in range(horizon):
    a = np.random.choice(env.actions)
    next_state, reward, _ = env.step(a)
    states.append(next_state)
    actions.append(a)
    model.add_state(state=tuple(next_state) , timestep=step)
    model.add_transition(tuple(current_state), a, tuple(next_state))
    model.add_reward(tuple(current_state-1), a, float(reward))
    current_state = next_state
model.finalize()

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

ax = env.plot()
xx = s0[:, 1] + 0.5
yy = s0[:, 0] + 0.5
ax.scatter(xx, yy, c=c0)

if args.video:
    plt.savefig(maze_file)

# Confirm that we're covering the state space relatively evenly
# np.histogram2d(states[:,0], states[:,1], bins=6)

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

x0 = sensor.observe(s0)
x1 = sensor.observe(s1)

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
    # 'L_fwd': args.L_fwd,
    'L_rat': args.L_rat,
    # 'L_fac': args.L_fac,
    'L_dis': args.L_dis,
    'L_ora': args.L_ora,
}

if args.type == 'markov':
    # discrete_cfg = {'groups':args.groups, 'n_embed':args.n_embed}
    fnet = FeatureNet(n_actions=4,
                      input_shape=x0.shape[1:],
                      n_latent_dims=args.latent_dims,
                      n_hidden_layers=1,
                      n_units_per_layer=32,
                      lr=args.learning_rate,
                      use_vq=args.use_vq,
                      use_proto=args.use_proto,
                      discrete_cfg=None, # TODO configure discrete_cfg
                      coefs=coefs)

elif args.type == 'genIK':
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


n_test_samples = 2000
test_s0 = s0[-n_test_samples:, :]
test_s1 = s1[-n_test_samples:, :]
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

def get_batch(x0, x1, a, s0, s1, batch_size=batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx]).float()
    tx1 = torch.as_tensor(x1[idx]).float()
    ta = torch.as_tensor(a[idx]).long()
    ti = torch.as_tensor(idx).long()

    sx0 = torch.as_tensor(s0[idx]).float()
    sx1 = torch.as_tensor(s1[idx]).float()

    return tx0, tx1, ta, idx, sx0, sx1


get_next_batch = (
    lambda: get_batch(   x0[:n_samples // 2, :],   x1[:n_samples // 2, :],    a[:n_samples // 2],   s0[:n_samples],  s1[:n_samples]   )   )







type1_evaluations = []

type2_evaluations = []


def test_rep(fnet, step,  ts0, ts1):
    with torch.no_grad():
        fnet.eval()
        if args.type== 'markov' :
            z0 = fnet.phi(test_x0)
            z1 = fnet.phi(test_x1)
            if fnet.use_vq:
                z0, zq_loss0, _ = fnet.vq_layer(z0)
                z1, zq_loss1, _ = fnet.vq_layer(z1)
                zq_loss = zq_loss0 + zq_loss1
                zq_loss = zq_loss.numpy().tolist()
            else:
                zq_loss = 0.
            # z1_hat = fnet.fwd_model(z0, test_a)
            # a_hat = fnet.inv_model(z0, z1)
            # yapf: disable
            loss_info = {
                'step': step,
                'L_inv': fnet.inverse_loss(z0, z1, test_a).numpy().tolist(),
                'L_coinv': fnet.contrastive_inverse_loss(z0, z1, test_a).numpy().tolist(),
                'L_fwd': 'NaN',  #fnet.compute_fwd_loss(z0, z1, z1_hat).numpy().tolist(),
                'L_rat': fnet.ratio_loss(z0, z1).numpy().tolist(),
                'L_dis': fnet.distance_loss(z0, z1).numpy().tolist(),
                'L_fac': 'NaN',  #fnet.compute_factored_loss(z0, z1).numpy().tolist(),
                'L_vq': zq_loss,#fnet.compute_entropy_loss(z0, z1, test_a).numpy().tolist(),
                'L': fnet.compute_loss(z0, z1, test_a, torch.zeros((2 * len(z0))), zq_loss).numpy().tolist(),
                'MI': MI(test_s0, z0.numpy()) / MI_max
            }
            # wandb.log(loss_info)

        elif args.type =='genIK':
            z0 = fnet.phi(test_x0)
            z1 = fnet.phi(test_x1)

            if fnet.use_vq:
                z0, zq_loss0, _ = fnet.vq_layer(z0)
                z1, zq_loss1, _ = fnet.vq_layer(z1)
                zq_loss = zq_loss0 + zq_loss1
                zq_loss = zq_loss.numpy().tolist()

                type1_err, type2_err = get_eval_error(z0, z1, ts0, ts1)


                type1_evaluations.append(type1_err)
                type2_evaluations.append(type2_err)

                if args.use_logger:
                    logger.record_type1_errors(type1_evaluations)
                    logger.record_type2_errors(type2_evaluations)
                    logger.save()

            # elif fnet.use_proto:
            #     z0, zq_loss0, _ = fnet.vq_layer(z0)
            #     z1, zq_loss1, _ = fnet.vq_layer(z1)
            #     zq_loss = zq_loss0 + zq_loss1
            #     zq_loss = zq_loss.numpy().tolist()
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
            # wandb.log(loss_info)

    json_str = json.dumps(loss_info)
    log.write(json_str + '\n')
    log.flush()

    text = '\n'.join([key + ' = ' + str(val) for key, val in loss_info.items()])

    results = [z0, z1, z1, test_a, test_a]
    return [r.numpy() for r in results] + [text], step, type1_err, type2_err



def get_eval_error (z0, z1, s0, s1):
    
    type1_err=0
    type2_err=0

    for i in range(z0.shape[0]):
        Z = z0[i] == z1[i]
        Z = Z.long()
        # Z_comp = Z[0] * Z[1] * Z[2] * Z[3]
        Z_comp = Z[0] * Z[1] 


        S = s0[i] == s1[i]
        S = S.long()
        S_comp = S[0] * S[1]

        if (1-Z_comp) and S_comp :
            #Error 2: Did not merge states which should be merged
            type2_err += 1

        if Z_comp and (1-S_comp):
            # Error 1: Merging states which should not be merged
            type1_err += 1

    return type1_err, type2_err

# def plot_rep_scatter(fnet, x):
#     with torch.no_grad():
#         z = fnet.phi(x)
#         if fnet.use_vq:
#             z, _, _ = fnet.vq_layer(z)
#     if fnet.n_latent_dims > 2:
#         fig = px.scatter_3d(x=z[:,0], y=z[:,1], z=z[:,2],
#                             color=z[:,2],
#                             color_continuous_scale='Viridis')
#     elif fnet.n_latent_dims == 2:
#         fig = px.scatter(x=z[:,0], y=z[:,1], color=z[:,1])
#     # wandb.log({'domain rep': fig})










#% ------------------ Run Experiment ------------------

#% ------------------ Run Experiment ------------------

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
        if frame_idx < 20:
            fnet.use_vq = False
        else:
            fnet.use_vq = args.use_vq
        '''
        without warm up
        '''
        fnet.use_vq = args.use_vq
        fnet.use_proto = args.use_proto
        fnet.train_batch(tx0, tx1, ta, tdist)

    # plot_rep_scatter(fnet, all_obs)
    test_results, step, type1_err, type2_err = test_rep(fnet, frame_idx * n_updates_per_frame,  ts0, ts1)

    if args.video:
        frame = repvis.update_plots(*test_results)
        # wandb.log({'original repo viz': wandb.Image(frame)})
        data.append(frame)

if args.video:
    imageio.mimwrite(video_filename, data, fps=15)
    # wandb.log({"video": wandb.Video(video_filename)})
    imageio.imwrite(image_filename, data[-1])

if args.save:
    fnet.phi.save('phi-{}'.format(args.seed), 'results/models/{}'.format(args.tag))
    if args.use_vq:
        torch.save(fnet.vq_layer.state_dict(), 'results/models/{}/vq.pt'.format(args.tag))
    elif args.use_proto:
        torch.save(fnet.proto.state_dict(), 'results/models/{}/proto.pt'.format(args.tag))



log.close()


# wandb_dir = wandb.run.dir[:-len('/files')]
# wandb.run.finish()

# # sync offline run to wandb
# if os.environ["WANDB_MODE"] == "offline":
#     os.system("wandb sync "+wandb_dir)
