#!/usr/bin/env python
# coding: utf-8

import os
import sys
import h5py
import copy 
import torch 
import random
import pygame
import numpy as np
import torch.nn as nn 
import numpy.random as npr
from datetime import datetime
import torch.nn.functional as F
from os.path import join, expanduser
sys.path.append("..")

from utility import *
import matplotlib.pyplot as plt 
from algorithms.networks.tranforms import *
import torchvision

load_aug_data = True  # load augmented data from disk?
save_aug_data = False  # save loaded files?


data_dir = join('..', 'experiments', 'inverted_pendulum', 'data_files')

if not load_aug_data:
    data_aug = DataAugmentor(data_dir=data_dir)
    observations, states = data_aug.get_augmented_tensor(rot_step=5)
else:
    loader = torch.load(join(data_dir, 'obs_state.pth'))
    observations, states = loader['observations'], loader['states']


# In[4]:


if save_aug_data:
    # save the augmented observations and states to disk
    # for faster future loading
    data_save = {'observations': observations,
    'states': states}
    torch.save(data_save, join(data_dir, 'obs_state.pth'))


# In[10]:


tr_ratio = int(.7*len(observations))
Xtr, Xte = observations[:tr_ratio], observations[tr_ratio:]

args = Bundle(dict(
                seed=123, NSamples=5000,  gpu=0, j=2, gamma=0.7,
                batch_size=10, test_batch_size=2,  lr=1e-3,
                weight_decay=1e-4,  momentum=.9, 
                num_epochs=50, best_loss=float("inf"), NRounds=200,
                loop_len=100, resume=False, use_lbfgs=False,
                log_interval=10, use_cuda=torch.cuda.is_available(),
            ))

dev = torch.device("cuda" if args.use_cuda else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)
train_kwargs = {'batch_size': args.batch_size,
                    'shuffle': True, 'num_workers': args.j}
test_kwargs = {'batch_size': args.test_batch_size,
                'shuffle': True, 'num_workers':  args.j}

data_kwargs = {'batch_size': args.batch_size,
                'shuffle': True, 'num_workers': 2}

if args.use_cuda:
    cuda_kwargs = {'num_workers': args.j,
                    'pin_memory': True,
                    'shuffle': True}
    data_kwargs.update(cuda_kwargs)
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

train_dataset = RobotDataSingle(Xtr, args.seed, dev)
train_loader  = torch.utils.data.DataLoader(train_dataset, **test_kwargs)

test_dataset  = RobotDataSingle(Xte, args.seed, device=dev)
test_loader   = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

def cast_and_normalise_images(images):
        """Convert images to floating point with the range [-0.5, 0.5]"""
        return (images.astype(np.float32) / 255.0) - 0.5

train_data_variance = torch.var(Xtr / 255.0)
print('train data variance: %s' % train_data_variance)


# In[19]:


# see: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb4
 
class ResidualStack(nn.Module):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
    super(ResidualStack, self).__init__()
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens

    self._layers = []
    for i in range(num_residual_layers):
      conv3 = nn.Conv2d(
          in_channels=num_residual_hiddens,
          out_channels=num_residual_hiddens,
          kernel_size=(3, 3),
          stride=(1, 1))
      conv1 = nn.Conv2d(
          in_channels=num_hiddens,
          out_channels=num_hiddens,
          kernel_size=(1, 1),
          stride=(1, 1))
      self._layers.append((conv3, conv1))

  def forward(self, inputs):
    h = inputs
    activation = nn.ReLU()
    for conv3, conv1 in self._layers:
      conv3_out = conv3(activation(h))
      conv1_out = conv1(activation(conv3_out))
      h += conv1_out
    return activation(h)  # Resnet V1 style


class Encoder(nn.Module):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
    super(Encoder, self).__init__()
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens

    self._enc_1 = nn.Conv2d(
        in_channels=self._num_hiddens // 2, 
        out_channels=self._num_hiddens // 2,
        kernel_size=(4, 4),
        stride=(2, 2))
    self._enc_2 = nn.Conv2d(
        in_channels=self._num_hiddens,
        out_channels=self._num_hiddens,
        kernel_size=(4, 4),
        stride=(2, 2))
    self._enc_3 = nn.Conv2d(
        in_channels=self._num_hiddens,
        out_channels=self._num_hiddens,
        kernel_size=(3, 3),
        stride=(1, 1),)
    self._residual_stack = ResidualStack(
        self._num_hiddens,
        self._num_residual_layers,
        self._num_residual_hiddens)

  def forward(self, x):
    activation = nn.ReLU()

    h = activation(self._enc_1(x))
    h = activation(self._enc_2(h))
    h = activation(self._enc_3(h))
    return self._residual_stack(h)


class Decoder(nn.Module):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
    super(Decoder, self).__init__()
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens

    self._dec_1 = nn.Conv2d(
        in_channels=self._num_hiddens,
        out_channels=self._num_hiddens,
        kernel_size=(3, 3),
        stride=(1, 1))
    self._residual_stack = ResidualStack(
        self._num_hiddens,
        self._num_residual_layers,
        self._num_residual_hiddens)
    self._dec_2 = nn.ConvTranspose2d(
        in_channels=self._num_hiddens,
        out_channels=self._num_hiddens // 2,
        kernel_size=(4, 4),
        stride=(2, 2))
    self._dec_3 = nn.ConvTranspose2d(
        in_channels=3,
        out_channels=3,
        kernel_size=(4, 4),
        stride=(2, 2))

  def forward(self, x):
    h = self._dec_1(x)
    h = self._residual_stack(h)
    h = F.relu(self._dec_2(h))
    x_recon = self._dec_3(h)

    return x_recon


class VQVAEModel(nn.Module):
  def __init__(self, encoder, decoder, vqvae, pre_vq_conv1,
               data_variance):
    super(VQVAEModel, self).__init__()
    self._encoder = encoder
    self._decoder = decoder
    self._vqvae = vqvae
    self._pre_vq_conv1 = pre_vq_conv1
    self._data_variance = data_variance

  def forward(self, inputs, is_training):
    z = self._pre_vq_conv1(self._encoder(inputs))
    vq_output = self._vqvae(z, is_training=is_training)
    x_recon = self._decoder(vq_output['quantize'])
    recon_error = torch.mean((x_recon - inputs) ** 2) / self._data_variance
    loss = recon_error + vq_output['loss']
    return {
        'z': z,
        'x_recon': x_recon,
        'loss': loss,
        'recon_error': recon_error,
        'vq_output': vq_output,
    }


# In[22]:


get_ipython().run_cell_magic('time', '', "\n# Set hyper-parameters.\nbatch_size = 32\nimage_size = 32\n\n# 100k steps should take < 30 minutes on a modern (>= 2017) GPU.\n# 10k steps gives reasonable accuracy with VQVAE on Cifar10.\nnum_training_updates = 10000\n\nnum_hiddens = 64\nnum_residual_hiddens = 32\nnum_residual_layers = 2\n# These hyper-parameters define the size of the model (number of parameters and layers).\n# The hyper-parameters in the paper were (For ImageNet):\n# batch_size = 128\n# image_size = 128\n# num_hiddens = 128\n# num_residual_hiddens = 32\n# num_residual_layers = 2\n\n# This value is not that important, usually 64 works.\n# This will not change the capacity in the information-bottleneck.\nembedding_dim = 64\n\n# The higher this value, the higher the capacity in the information bottleneck.\nnum_embeddings = 512\n\n# commitment_cost should be set appropriately. It's often useful to try a couple\n# of values. It mostly depends on the scale of the reconstruction cost\n# (log p(x|z)). So if the reconstruction cost is 100x higher, the\n# commitment_cost should also be multiplied with the same amount.\ncommitment_cost = 0.25\n\n# Use EMA updates for the codebook (instead of the Adam optimizer).\n# This typically converges faster, and makes the model less dependent on choice\n# of the optimizer. In the VQ-VAE paper EMA updates were not used (but was\n# developed afterwards). See Appendix of the paper for more details.\nvq_use_ema = True\n\n# This is only used for EMA updates.\ndecay = 0.99\n\nlearning_rate = 3e-4\n\n# Build modules.\nencoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)\ndecoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens)")


# In[ ]:



pre_vq_conv1 = snt.Conv2D(output_channels=embedding_dim,
    kernel_shape=(1, 1),
    stride=(1, 1),
    name="to_vq")

if vq_use_ema:
  vq_vae = snt.nets.VectorQuantizerEMA(
      embedding_dim=embedding_dim,
      num_embeddings=num_embeddings,
      commitment_cost=commitment_cost,
      decay=decay)
else:
  vq_vae = snt.nets.VectorQuantizer(
      embedding_dim=embedding_dim,
      num_embeddings=num_embeddings,
      commitment_cost=commitment_cost)
  
model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1,
                   data_variance=train_data_variance)

optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

@tf.function
def train_step(data):
  with tf.GradientTape() as tape:
    model_output = model(data['image'], is_training=True)
  trainable_variables = model.trainable_variables
  grads = tape.gradient(model_output['loss'], trainable_variables)
  optimizer.apply(grads, trainable_variables)

  return model_output

train_losses = []
train_recon_errors = []
train_perplexities = []
train_vqvae_loss = []

for step_index, data in enumerate(train_dataset):
  train_results = train_step(data)
  train_losses.append(train_results['loss'])
  train_recon_errors.append(train_results['recon_error'])
  train_perplexities.append(train_results['vq_output']['perplexity'])
  train_vqvae_loss.append(train_results['vq_output']['loss'])

  if (step_index + 1) % 100 == 0:
    print('%d train loss: %f ' % (step_index + 1,
                                   np.mean(train_losses[-100:])) +
          ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
          ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
          ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])))
  if step_index == num_training_updates:
    break


# In[ ]:



args = Bundle(dict(
    seed=123, NSamples=5000,  gpu=0, j=2, gamma=0.7,
        batch_size=1024, test_batch_size=1000,  lr=1e-7,
    weight_decay=1e-4,  momentum=.9, trainWith='sup',
    num_epochs=50, best_loss=float("inf"), NRounds=200,
    loop_len=100, resume=False, use_lbfgs=False,
    log_interval=10, use_cuda=torch.cuda.is_available(),
    ))
dev = torch.device("cuda" if args.use_cuda else "cpu")

# if torch.cuda.is_available():
#     torch.set_default_tensor_type(torch.cuda.DoubleTensor)
# else:
torch.set_default_tensor_type(torch.DoubleTensor)
train_kwargs = {'batch_size': args.batch_size,
                    'shuffle': True, 'num_workers': args.j}
test_kwargs = {'batch_size': args.test_batch_size,
                    'shuffle': True, 'num_workers':  args.j}

data_kwargs = {'batch_size': args.batch_size,
                    'shuffle': True, 'num_workers': 2}
if args.use_cuda:
    cuda_kwargs = {'num_workers': args.j,
                    'pin_memory': True,
                    'shuffle': True}
    data_kwargs.update(cuda_kwargs)
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


# In[ ]:




