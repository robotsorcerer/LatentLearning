import sys
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from dvq_vae.data.invpend import InvertedPendulumData
from pytorch_lightning.callbacks import ModelCheckpoint
from dvq_vae.vqvae import *
from algorithms.networks.housekeeping.augment import get_obs_states
from utility import *
from absl import flags, app
sys.path.append('../')

args = Bundle(dict(
                seed=123, NSamples=5000,  gpu=0, j=2, gamma=0.7,
                batch_size=128, test_batch_size=2,  lr=1e-3,
                weight_decay=1e-4,  momentum=.9, gpus=1,
                num_epochs=50, best_loss=float("inf"), num_workers=8,
                loop_len=100, resume=False, use_lbfgs=False,
                log_interval=10, use_cuda=torch.cuda.is_available(),
            ))

FLAGS = flags.FLAGS


def main():
    pl.seed_everything(123)
    # -------------------------------------------------------------------------
    # arguments...
    parser = ArgumentParser()
    # training related
    parser = pl.Trainer.add_argparse_args(parser)
    # model related
    parser = VQVAE.add_model_specific_args(parser)
    # dataloader related
    parser.add_argument("--data_dir", type=str, default='/opt/invpend')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    # done!
    args = parser.parse_args()

    data = InvertedPendulumData(args)
    model = VQVAE(args)

    # annealing schedules for lots of constants
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val_recon_loss', mode='min'))
    callbacks.append(DecayLR())
    if args.vq_flavor == 'gumbel':
        callbacks.extend([DecayTemperature(), RampBeta()])
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps=3000000)

    trainer.fit(model, data)

if __name__ == "__main__":
    main()    