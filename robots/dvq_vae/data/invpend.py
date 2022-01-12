__all__ = ["InvertedPendulumData"]

from torch.utils.data import DataLoader
from torchvision import transforms as T
import pytorch_lightning as pl

from utility import Bundle
from os.path import join
from algorithms.networks.housekeeping import get_obs_states


loadargs = Bundle(dict(load_aug_data = True, im_size=(128, 128),
					data_dir=join('experiments', 'inverted_pendulum', 'data_files'),
					rot_step=15, save_aug_data = False))  # save loaded files?
observations, states = get_obs_states(loadargs)

tr_ratio = int(.7*len(observations))
Xtr, Xte = observations[:tr_ratio], observations[tr_ratio:]


class InvertedPendulumData(pl.LightningDataModule):
	""" returns observations in floats in range [0,1] """
	def __init__(self, args):
		print('self.hparams ', args)
		super().__init__()
		self.hparams = args
		print('self.hparams ', self.hparams)

	def train_dataloader(self):
		dataloader = DataLoader(
			Xtr,
			batch_size=self.hparams.batch_size,
			num_workers=self.hparams.num_workers,
			drop_last=True,
			pin_memory=True,
			shuffle=True,
		)

		return dataloader

	def val_dataloader(self):
		dataloader = DataLoader(
			Xte,
			batch_size=self.hparams.batch_size,
			num_workers=self.hparams.num_workers,
			drop_last=True,
			pin_memory=True,
		)
		return dataloader

	def test_dataloader(self):
		return self.val_dataloader()
