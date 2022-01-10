import torch
import logging
import argparse 
from .augment import DataAugmentor
from os.path import join, expanduser
from algorithms.networks.tranforms import *

from utility import *


parser = argparse.ArgumentParser(description='Latent States Learner')
parser.add_argument('--load_aug_data', '-la', action='store_true', default=False, help='compute trajectory?')
parser.add_argument('--save_aug_data', '-sa', action='store_false', help='silent debug print outs' )
parser.add_argument('--visualize', '-vz', action='store_true', default=True, help='visualize level sets?' )
parser.add_argument('--pause_time', '-pz', type=float, default=5e-3, help='pause time between successive updates of plots' )
args = parser.parse_args()

print('args: ', args)

if args.silent:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
else:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
# Turn off pyplot's spurious dumps on screen
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

def get_data(args):
    data_dir = join('experiments', 'inverted_pendulum', 'data_files')

    if not args.load_aug_data:
        data_aug = DataAugmentor(data_dir=data_dir)
        observations, states = data_aug.get_augmented_tensor(rot_step=5, im_size=(128, 128))
    else:
        loader = torch.load(join(data_dir, 'obs_state.pth'))
        observations, states = loader['observations'], loader['states']

    if args.save_aug_data:
        # save the augmented observations and states to disk
        # for faster future loading
        data_save = {'observations': observations,
                     'states': states}
        
        save_path = join(data_dir, 'obs_state.pth')
        torch.save(data_save, save_path)        

    return observations, states
