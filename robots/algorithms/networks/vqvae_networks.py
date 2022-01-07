import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualStack(nn.Module):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
    super(ResidualStack, self).__init__()
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens

    self._layers = []
    for i in range(num_residual_layers):
      conv3 = nn.Conv2D(
          out_channels=num_residual_hiddens,
          kernel_size=(3, 3),
          stride=(1, 1))
      conv1 = nn.Conv2D(
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

    self._enc_1 = nn.Conv2D(
        in_channels=
        out_channels=self._num_hiddens // 2,
        kernel_size=(4, 4),
        stride=(2, 2))
    self._enc_2 = nn.Conv2D(
        in_channels=
        out_channels=self._num_hiddens,
        kernel_size=(4, 4),
        stride=(2, 2))
    self._enc_3 = nn.Conv2D(
        in_channels=
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

    self._dec_1 = nn.Conv2D(
        in_channels=
        out_channels=self._num_hiddens,
        kernel_size=(3, 3),
        stride=(1, 1))
    self._residual_stack = ResidualStack(
        self._num_hiddens,
        self._num_residual_layers,
        self._num_residual_hiddens)
    self._dec_2 = nn.ConvTranspose2d(
        in_channels=
        out_channels=self._num_hiddens // 2,
        output_shape=None,
        kernel_size=(4, 4),
        stride=(2, 2))
    self._dec_3 = nn.ConvTranspose2d(
        in_channels=
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
