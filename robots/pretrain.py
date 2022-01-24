import sys
import torch
from utility import *
sys.path.append('../')
from os.path import join
from dvq_vae.vqvae import *
from absl import flags, app
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from torchvision import transforms as T
from algorithms.networks.housekeeping.augment import get_obs_states
torch.set_default_tensor_type(torch.FloatTensor)

args = Bundle(dict(
                seed=123, NSamples=5000,  gpu=0, j=2, gamma=0.7,
                batch_size=128, test_batch_size=2,  lr=1e-3,
                weight_decay=1e-4,  momentum=.9, gpus=1,
                num_epochs=50, best_loss=float("inf"), num_workers=6,
                loop_len=100, resume=False, use_lbfgs=False,
                log_interval=10, use_cuda=torch.cuda.is_available(),
            ))

loadargs = Bundle(dict(load_aug_data = True, im_size=(128, 128),
					data_dir=join('experiments', 'inverted_pendulum', 'data_files'),
					rot_step=15, save_aug_data = False))  # save loaded files?
observations, states = get_obs_states(loadargs)

tr_ratio = int(.7*len(observations))
Xtr, Xte = observations[:tr_ratio].double(), observations[tr_ratio:].double()


class InvertedPendulumData(pl.LightningDataModule):
	""" returns observations in floats in range [0,1] """
	def __init__(self, args):
		super().__init__()

	def train_dataloader(self):
		dataloader = DataLoader(
			Xtr,
			batch_size=128, #self.hparams.batch_size,
			num_workers=6, #self.hparams.num_workers,
			drop_last=True,
			pin_memory=True,
			shuffle=True,
		)
		return dataloader

	def val_dataloader(self):
		dataloader = DataLoader(
			Xte,
			batch_size=128, #self.hparams.batch_size,
			num_workers=6, #self.hparams.num_workers,
			drop_last=True,
			pin_memory=True,
		)
		return dataloader

	def test_dataloader(self):
		return self.val_dataloader()

def main(args):
    pl.seed_everything(args.seed)
    # -------------------------------------------------------------------------
    # arguments...
    parser = ArgumentParser()
    # training related
    parser = pl.Trainer.add_argparse_args(parser)
    # model related
    parser = VQVAE.add_model_specific_args(parser)
    # dataloader related
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=args.num_workers)
    parser.add_argument("--expt_dir", type=str, default="experiments/inverted_pendulum/checkpoints/")
    # done!
    args = parser.parse_args()

    data = InvertedPendulumData(args)
    model = VQVAE(args)

    # annealing schedules for lots of constants
    callbacks = []
    callbacks.append(ModelCheckpoint(dirpath=args.expt_dir, monitor='val_recon_loss', mode='min'))
    callbacks.append(DecayLR())
    
    if args.vq_flavor == 'gumbel':
        callbacks.extend([DecayTemperature(), RampBeta()])
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps=3000000)

    trainer.fit(model, data)

if __name__ == "__main__":
    main(args)    