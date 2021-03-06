import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import pandas as pd
import seaborn as sns
import seeding
from tqdm import tqdm
import time
from collections import deque
from datetime import datetime
from models.nnutils import Reshape
from models.nullabstraction import NullAbstraction
from models.phinet import PhiNet
from models.vq_layer_kmeans import Quantize
from repvis import RepVisualization, CleanVisualization
from agents.randomagent import RandomAgent
from agents.dqnagent import DQNAgent
from visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld, MazeWorld, SpiralWorld, LoopWorld
from visgrid.utils import get_parser
from visgrid.sensors import *
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

parser = get_parser()

# yapf: disable
parser.add_argument('-a','--agent', type=str, required=True,
                    choices=['random','dqn'], help='Type of agent to train')

parser.add_argument('-n','--n_trials', type=int, default=300,
                    help='Number of trials')

parser.add_argument('-e','--n_episodes', type=int, default=100,
                    help='Number of episodes per trial')

parser.add_argument('-m','--max_steps', type=int, default=1000,
                    help='Maximum number of steps per episode')

parser.add_argument('-r','--rows', type=int, default=6,
                    help='Number of gridworld rows')

parser.add_argument('-c','--cols', type=int, default=6,
                    help='Number of gridworld columns')

parser.add_argument('-w', '--walls', type=str, default='empty', choices=['empty', 'maze', 'spiral', 'loop'],
                    help='The wall configuration mode of gridworld')


parser.add_argument('-b','--batch_size', type=int, default=16,
                    help='Number of experiences to sample per batch')

parser.add_argument('-l','--latent_dims', type=int, default=256,

                    help='Number of latent dimensions to use for representation')
parser.add_argument('-lr','--learning_rate', type=float, default=0.003,
                    help='Learning rate for Adam optimizer')

parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')

parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')

parser.add_argument('--phi_path', type=str,
                    help='Load an existing abstraction network by tag')

parser.add_argument('--no_phi', action='store_true',
                    help='Turn off abstraction and just use observed state; i.e. ??(x)=x')

parser.add_argument('--train_phi', action='store_true',
                    help='Allow simultaneous training of abstraction')

parser.add_argument('--no_sigma', action='store_true',
                    help='Turn off sensors and just use true state; i.e. x=s')

parser.add_argument('--one_hot', action='store_true',
                    help='Bypass sensor and use one-hot representation instead')

parser.add_argument('--save', action='store_true',
                    help='Save final network weights')

parser.add_argument('-v','--video', action='store_true',
                    help='Show video of agent training')

parser.add_argument('--xy_noise', action='store_true',
                    help='Add truncated gaussian noise to x-y positions')

parser.add_argument('--rearrange_xy', action='store_true',
                    help='Rearrange discrete x-y positions to break smoothness')

parser.add_argument('--use_vq', action='store_true',
                    help='Use VQ layer after the phi network')

# yapf: enable
args = parser.parse_args()
if args.train_phi and args.no_phi:
    assert False, '--no_phi and --train_phi are mutually exclusive'

if args.one_hot and args.no_sigma:
    assert False, '--one_hot and --no_sigma are mutually exclusive'

if args.video:
    import matplotlib.pyplot as plt

if args.walls in ['empty', 'spiral', 'loop']:
    maze_dir = 'maze_' + args.walls
else:
    maze_dir = 'maze' + str(args.seed)

log_dir = 'results/scores/{}/pretrain_3k/'.format(maze_dir) + str(args.tag)
os.makedirs(log_dir, exist_ok=True)
log = open(log_dir + '/scores-{}-{}.txt'.format(args.agent, args.seed), 'w')

#%% ------------------ Define MDP ------------------
if args.walls == 'maze':
    env = MazeWorld.load_maze(rows=args.rows, cols=args.cols, seed=args.seed)
elif args.walls == 'spiral':
    env = SpiralWorld(rows=args.rows, cols=args.cols)
elif args.walls == 'loop':
    env = LoopWorld(rows=args.rows, cols=args.cols)
else:
    env = GridWorld(rows=args.rows, cols=args.cols)
gamma = 0.99

#%% ------------------ Define sensor ------------------
sensor_list = []

if args.xy_noise:
    sensor_list.append(NoisySensor(sigma=0.2, truncation=0.4))


if args.rearrange_xy:
    sensor_list.append(RearrangeXYPositionsSensor((env._rows, env._cols)))

if not args.no_sigma:
    if args.one_hot:
        sensor_list += [
            OffsetSensor(offset=(0.5, 0.5)),
            ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=1),
        ]
    else:
        sensor_list += [
            OffsetSensor(offset=(0.5, 0.5)),
            NoisySensor(sigma=0.05),
            ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=3),
            # ResampleSensor(scale=2.0),
            BlurSensor(sigma=0.6, truncate=1.),
            NoisySensor(sigma=0.01)
        ]
sensor = SensorChain(sensor_list)



#%% ------------------ Define abstraction ------------------
if args.no_phi:
    phinet = NullAbstraction(-1, args.latent_dims)
else:
    x0 = sensor.observe(env.get_state())
    phinet = PhiNet(input_shape=x0.shape,
                    n_latent_dims=args.latent_dims,
                    n_hidden_layers=1,
                    n_units_per_layer=32)


    if args.phi_path:
        modelfile = 'results/models/{}/phi-{}_latest.pytorch'.format(args.phi_path, args.seed)
        phinet.load(modelfile)

        if args.use_vq:
            vq_layer = Quantize(args.latent_dims, 50, 1)
            vq_path = 'results/models/{}/vq.pt'.format(args.phi_path)

            vq_layer.load_state_dict(
                torch.load(vq_path)
            )
        else:
            vq_layer = None

        # visualize the features to ensure the representation is rightly loaded
        vis_features = False

        if vis_features:
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
                    z1, zq_loss1, _ = vq_layer(z1)
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
                     lr=args.learning_rate,
                     batch_size=args.batch_size,
                     train_phi=args.train_phi,
                     gamma=gamma,
                     decay_period=args.n_episodes * args.max_steps * 0.6,
                     vq_layer=vq_layer,
                     epsilon=0.1)  # set a larger epsilon
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

for trial in tqdm(range(args.n_trials), desc='trials'):
    if args.walls in ['empty']:
        goal_pos = None
    else:
        goal_pos = [2, 3]
    env.reset_goal(goal_pos=goal_pos)
    agent.reset()
    total_reward = 0
    total_steps = 0
    losses = []
    rewards = []
    value_fn = []
    reward_lst = deque(maxlen=500)


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
            rewards.append(r)

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
        if episode % 50 == 0:
            eval_agent(goal_pos, env, args, sensor, agent, tb_logger, episode)

        total_steps += step
        score_info = {
            'trial': trial,
            'episode': episode,
            'reward': sum(ep_rewards),
            'total_reward': total_reward,
            'total_steps': total_steps,
            'steps': step
        }
        reward_lst.append(sum(ep_rewards))
        wd_size = 50
        mean200 = np.mean(list(reward_lst)[-wd_size:])
        success = np.sum(np.array(reward_lst)[-wd_size:] > -args.max_steps)
        success = success / wd_size
        # record in tensorboard
        tb_logger.add_scalar("train/reward", mean200, episode)
        tb_logger.add_scalar("train/success", success, episode)
        json_str = json.dumps(score_info)
        log.write(json_str + '\n')
        log.flush()
print('\n\n')

if args.save:
    agent.q.save('qnet-{}'.format(args.seed), 'results/models/{}'.format(args.tag))
