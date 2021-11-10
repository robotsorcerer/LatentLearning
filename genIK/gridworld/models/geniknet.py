from collections import defaultdict
import numpy as np
import torch
import torch.nn

from .nnutils import Network
from .phinet import PhiNet
from .invnet import InvNet
from .fwdnet import FwdNet
from .contrastivenet import ContrastiveNet
from .invdiscriminator import InvDiscriminator
from .vq_layer_kmeans import Quantize
from .proto import Proto
from .proto_utils import soft_update_params


class GenIKNet(Network):
    def __init__(self,
                 n_actions,
                 input_shape=2,
                 n_latent_dims=4,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 lr=0.001,
                 use_vq=None,
                 discrete_cfg=None, #groups and n_embed
                 use_proto=None,
                 phi_target_tau=0.05,
                 coefs=None):
        super().__init__()
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.lr = lr
        self.coefs = defaultdict(lambda: 1.0)
        if coefs is not None:
            for k, v in coefs.items():
                self.coefs[k] = v

        #PhiNet - outputs features
        self.phi = PhiNet(input_shape=input_shape,
                          n_latent_dims=n_latent_dims,
                          n_units_per_layer=n_units_per_layer,
                          n_hidden_layers=n_hidden_layers)

        self.use_vq = use_vq
        self.use_proto = use_proto

        # add a VQ layer
        if self.use_vq:
            assert discrete_cfg is not None
            self.vq_layer = Quantize(n_latent_dims, discrete_cfg['n_embed'], discrete_cfg['groups'])

        if self.use_proto:
            # assert discrete_cfg is not None
            self.phi_target = PhiNet(input_shape=input_shape,
                                     n_latent_dims=n_latent_dims,
                                     n_units_per_layer=n_units_per_layer,
                                     n_hidden_layers=n_hidden_layers)
            self.phi_target.load_state_dict(self.phi.state_dict())
            self.phi_target_tau = phi_target_tau
            self.proto = Proto(n_latent_dims, 16, 0.1, 10, 3, 3) # these values may be given in args or config

        # self.fwd_model = FwdNet(n_actions=n_actions, n_latent_dims=n_latent_dims, n_hidden_layers=n_hidden_layers, n_units_per_layer=n_units_per_layer)
        self.inv_model = InvNet(n_actions=n_actions,
                                n_latent_dims=n_latent_dims,
                                n_units_per_layer=n_units_per_layer,
                                n_hidden_layers=n_hidden_layers)
        self.inv_discriminator = InvDiscriminator(n_actions=n_actions,
                                                  n_latent_dims=n_latent_dims,
                                                  n_units_per_layer=n_units_per_layer,
                                                  n_hidden_layers=n_hidden_layers)
        self.discriminator = ContrastiveNet(n_latent_dims=n_latent_dims,
                                            n_hidden_layers=1,
                                            n_units_per_layer=n_units_per_layer)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def inverse_loss(self, z0, z1, a):
        if self.coefs['L_inv'] == 0.0:
            return torch.tensor(0.0)
        a_hat = self.inv_model(z0, z1)
        return self.cross_entropy(input=a_hat, target=a)

    def contrastive_inverse_loss(self, z0, z1, a):
        if self.coefs['L_coinv'] == 0.0:
            return torch.tensor(0.0)
        N = len(z0)
        # shuffle next states
        idx = torch.randperm(N)

        a_neg = torch.randint_like(a, low=0, high=self.n_actions)

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_extended = torch.cat([z1, z1], dim=0)
        a_pos_neg = torch.cat([a, a_neg], dim=0)
        is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0)

        # Compute which ones are fakes
        fakes = self.inv_discriminator(z0_extended, z1_extended, a_pos_neg)
        contrastive_loss = self.bce_loss(input=fakes, target=is_fake.float())

        return contrastive_loss

    def get_protos(self, x0, x1):
        assert self.use_proto
        return self.proto.get_protos(self.phi, self.phi_target, x0, x1)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict_a(self, z0, z1):
        raise NotImplementedError
        # a_logits = self.inv_model(z0, z1)
        # return torch.argmax(a_logits, dim=-1)

    def compute_loss(self, z0, z1, a, d):
        loss = self.coefs['L_coinv'] * self.contrastive_inverse_loss(z0, z1, a)  # zero
        loss += self.coefs['L_inv'] * self.inverse_loss(z0, z1, a)  # inverse model: 1

        return loss

    def train_batch(self, x0, x1, a, d):
        self.train()
        self.optimizer.zero_grad()
        z0 = self.phi(x0)
        z1 = self.phi(x1)

        if self.use_vq:
            z0, zq_loss0, z_discrete0  = self.vq_layer(z0)
            z1, zq_loss1, z_discrete1 = self.vq_layer(z1)
            zq_loss = zq_loss0 + zq_loss1

            
        elif self.use_proto:
            # TODO: for now no visual data augmentation is used (proto-rl does), can easily add it
            z0, z1, zq_loss = self.get_protos(x0, x1)
        else:
            zq_loss = 0
        # z1_hat = self.fwd_model(z0, a)
        loss = self.compute_loss(z0, z1, a, d)
        loss += zq_loss
        loss.backward()
        self.optimizer.step()

        # ema update of phi_target
        if self.use_proto:
            soft_update_params(self.phi, self.phi_target, self.phi_target_tau)

        return loss
