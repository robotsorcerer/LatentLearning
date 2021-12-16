import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.util import view_as_windows
import random

from noisy_state_abstractions.algorithms.dqn_infomax.utils import init, compute_network_output_size, rollout, make_one_hot

"""
Fully-connected networks
- FC vanilla
- FC Rainbow (expects advantage and value features)
"""

class Action_net(nn.Module):
    def __init__(self,n_actions):
        super(Action_net,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_actions, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        return self.layers(x)

class FC(nn.Module): # input is obs_space
    def __init__(self, obs_space, num_outputs, options):
        super(FC, self).__init__()
        layers = []
        layers.append(SlimFC(obs_space, num_outputs, initializer=nn.init.xavier_uniform_))
        self.layers = nn.Sequential(*layers)

    def forward(self, input_dict):
        q = self.layers(input_dict["output_bottom"])
        outputs = {'output_top':q}
        return outputs

class FCRainbow(nn.Module): # input is obs_space
    def __init__(self, obs_space, num_outputs, options):
        super(FCRainbow, self).__init__()
        self.num_actions = num_outputs
        self.atoms = options.get('num_atoms')
        self.hiddens = options.get('fcnet_hiddens')
        noisy = options.get('noisy')
        self.log = options.get('log')

        fc = NoisyLinear if noisy else lambda inp,out:SlimFC(inp,out,initializer=nn.init.xavier_uniform_)

        self.fc_z_v = fc(512, self.atoms)
        self.fc_z_a = fc(512, self.num_actions * self.atoms)


    def forward(self, input_dict):
        features_v,features_a = input_dict["output_bottom"]
        v = self.fc_z_v(features_v)  # Value stream
        a = self.fc_z_a(features_a)  # Advantage stream

        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.num_actions, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        # Use log softmax for numerical stability
        log_q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        outputs = {'output_top':q,"log_q":log_q}
        return outputs

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name and module == NoisyLinear:
                module.reset_noise()

"""
Encoder networks (Cnn-based)
"""

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, input_batch):
        features = self.global_encoder(self.local_encoder(input_batch))
        return features

"""
infoNCE encoders for different inputs
"""

class CURL(Encoder): # input is (84,84,4)
    def __init__(self, obs_space, num_outputs, options):
        """
        3 Conv layers with 1 FC
        IM encoder is inserted after 3rd conv, and has 1x1 kernel
        """

        super().__init__()
        (w, h, in_channels) = obs_space.shape

        action_dim = 64
        rkhs_dim = 512
        init_fn = lambda m: init(m,
                               lambda x:nn.init.orthogonal_(x,gain=nn.init.calculate_gain('relu')),
                               #lambda x:torch.nn.init.kaiming_uniform_(x,a=0,mode='fan_in',nonlinearity='relu'),
                               lambda x: nn.init.constant_(x, 0))

        self.out_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64
        self.fc_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64

        self.conv1 = init_fn( nn.Conv2d(in_channels, 32, [8,8], 4) )
        self.conv2 = init_fn( nn.Conv2d(32, 64, [4,4], 2) )
        self.conv3 = init_fn( nn.Conv2d(64, 64, [3,3], 1) )
        self.flatten = Flatten()
        self.fc_1 = init_fn( nn.Linear(self.fc_channels, 512) )

        self.psi_q = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)
        self.psi_k = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)

        self.psi_local_LL_t_p_1 = ResBlock_conv(64,rkhs_dim,rkhs_dim,init_fn)

        self.layers = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU(),self.flatten,self.fc_1,nn.ReLU()])
        self.convs  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])

        self.global_encoder = nn.Sequential(*[self.flatten,self.fc_1])
        self.local_encoder  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])
        # self.action_encoder = OneHot_fc(num_outputs,action_dim,action_dim,init_fn)

    def forward(self, input_batch):
        input_batch = self.augment(input_batch)
        features = self.global_encoder(self.local_encoder(input_batch))
        return features

    def augment(self, x):
        x = x.cpu().numpy()
        x = self.curl_random_crop(x,60)
        x = self.color_jitter(x,0.4,0.4)
        x = torch.FloatTensor(x)
        if torch.cuda.is_available():
            x = x.cuda()
        return x

    def curl_random_crop(self, imgs, out):
        """
        Vectorized random crop
        args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
        """
        shape = imgs.shape
        # n: batch size.
        n = imgs.shape[0]
        img_size = imgs.shape[-1] # e.g. 100
        crop_max = img_size - out
        imgs = np.transpose(imgs, (0, 2, 3, 1))
        w1 = np.random.randint(0, crop_max, n)
        h1 = np.random.randint(0, crop_max, n)
        # creates all sliding window
        # combinations of size (out)
        windows = view_as_windows(imgs, (1, out, out, 1))[..., 0,:,:, 0]
        # selects a random window# for each batch element
        cropped = windows[np.arange(n), w1, h1]
        cropped = np.resize(cropped,shape)
        return cropped

    def color_jitter(self,x,brightness,contrast):
        # alpha = 1.0 + random.uniform(-brightness, brightness)
        # x *= alpha

        alpha = 1.0 + random.uniform(-contrast, contrast)
        coef = np.array([[[0.299, 0.587, 0.114]]]).reshape(1,3,1,1)
        gray = x * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        x *= alpha
        x += gray
        return x

class Mnih_84x84_vae(nn.Module): 
    def __init__(self, obs_space, num_outputs, options):
        """
        3 Conv layers with 1 FC
        IM encoder is inserted after 3rd conv, and has 1x1 kernel
        """

        super().__init__()
        (w, h, in_channels) = obs_space.shape

        init_fn = lambda m: init(m,
                               lambda x:nn.init.orthogonal_(x,gain=nn.init.calculate_gain('relu')),
                               #lambda x:torch.nn.init.kaiming_uniform_(x,a=0,mode='fan_in',nonlinearity='relu'),
                               lambda x: nn.init.constant_(x, 0))

        self.out_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64
        self.fc_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64

        self.conv1 = init_fn( nn.Conv2d(in_channels, 32, [8,8], 4) )
        self.conv2 = init_fn( nn.Conv2d(32, 64, [4,4], 2) )
        self.conv3 = init_fn( nn.Conv2d(64, 64, [3,3], 1) )
        self.flatten = Flatten()
        self.fc_1 = init_fn( nn.Linear(self.fc_channels, 512) )


        self.fc1 = nn.Linear(3456, 16)
        self.fc2 = nn.Linear(3456, 16)
        self.fc3 = nn.Linear(16, 3456)

        self.deconv3 = init_fn( nn.ConvTranspose2d(3456, 64, kernel_size=[8,6], stride=1) )
        self.deconv2 = init_fn( nn.ConvTranspose2d(64, 32, kernel_size=[10,8], stride=2) )
        self.deconv1 = init_fn( nn.ConvTranspose2d(32, in_channels, kernel_size=[12,12], stride=4) )

        self.decoder = nn.Sequential(
            UnFlatten(),
            self.deconv3,
            nn.ReLU(),
            self.deconv2,
            nn.ReLU(),
            self.deconv1,
            nn.Sigmoid()
        )

        self.layers = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU(),self.flatten,self.fc_1,nn.ReLU()])
        self.convs  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])

        self.flat_convs = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU(),self.flatten])

        self.global_encoder = nn.Sequential(*[self.flatten,self.fc_1])
        self.local_encoder  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])

    def reparameterize(self, mu, logvar, device):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        eps = torch.randn(*mu.size()).to(device)
        z = mu + std * eps
        return z
    
    def bottleneck(self, h, device):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar, device)
        return z, mu, logvar

    def encode(self, x, device):
        h = self.flat_convs(x)
        z, mu, logvar = self.bottleneck(h, device)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z
    
    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()

    def forward(self, input_batch):
        features = self.global_encoder(self.local_encoder(input_batch))
        return features

    def predict(self, x, device):
        z, mu, logvar = self.encode(x, device)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

"""
########################################################
"""

class infoNCE_Mnih_84x84_action_FILM(Encoder): # input is (84,84,4)
    def __init__(self, obs_space, num_outputs, options):
        """
        3 Conv layers with 1 FC
        IM encoder is inserted after 3rd conv, and has 1x1 kernel
        """

        super().__init__()
        (w, h, in_channels) = obs_space.shape

        action_dim = num_outputs
        rkhs_dim = 512
        layernorm = True
        init_fn = lambda m: init(m,
                               lambda x:nn.init.orthogonal_(x,gain=nn.init.calculate_gain('relu')),
                               #lambda x:torch.nn.init.kaiming_uniform_(x,a=0,mode='fan_in',nonlinearity='relu'),
                               lambda x: nn.init.constant_(x, 0))

        self.out_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64
        self.fc_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64

        self.conv1 = init_fn( nn.Conv2d(in_channels, 32, [8,8], 4) )
        self.conv2 = init_fn( nn.Conv2d(32, 64, [4,4], 2) )
        self.conv3 = init_fn( nn.Conv2d(64, 64, [3,3], 1) )
        self.flatten = Flatten()
        self.fc_1 = init_fn( nn.Linear(self.fc_channels, 512) )

        self.psi_local_LL_t = ResBlock_conv(64+action_dim,rkhs_dim,rkhs_dim,init_fn)
        self.psi_local_LL_t_p_1 = ResBlock_conv(64,rkhs_dim,rkhs_dim,init_fn)
        self.psi_local_LG = ResBlock_conv(64+action_dim,rkhs_dim,rkhs_dim,init_fn)
        self.psi_local_GL = ResBlock_conv(64,rkhs_dim,rkhs_dim,init_fn)

        self.psi_global_LG = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)
        self.psi_global_GL = ResBlock_fc_FILM(512,rkhs_dim,rkhs_dim,action_dim,layernorm,init_fn)
        self.psi_global_GG_t = ResBlock_fc_FILM(512,rkhs_dim,rkhs_dim,action_dim,layernorm,init_fn)
        self.psi_global_GG_t_p_1 = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)

        self.layers = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU(),self.flatten,self.fc_1,nn.ReLU()])
        self.convs  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])

        self.global_encoder = nn.Sequential(*[self.flatten,self.fc_1])
        self.local_encoder  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])
        self.action_encoder = OneHot_fc(action_dim,action_dim,action_dim,init_fn)


class infoNCE_Mnih_84x84(Encoder): # input is (84,84,4)
    def __init__(self, obs_space, num_outputs, options):
        """
        3 Conv layers with 1 FC
        IM encoder is inserted after 3rd conv, and has 1x1 kernel
        """

        super().__init__()
        (w, h, in_channels) = obs_space.shape

        rkhs_dim = 128
        init_fn = lambda m: init(m,
                               lambda x:torch.nn.init.kaiming_uniform_(x,a=0,mode='fan_in',nonlinearity='relu'),
                               lambda x: nn.init.constant_(x, 0))

        self.out_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64
        self.fc_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64

        self.conv1 = init_fn( nn.Conv2d(in_channels, 32, [8,8], 4) )
        self.conv2 = init_fn( nn.Conv2d(32, 64, [4,4], 2) )
        self.conv3 = init_fn( nn.Conv2d(64, 64, [3,3], 1) )
        self.flatten = Flatten()
        self.fc_1 = init_fn( nn.Linear(self.fc_channels, 512) )

        self.psi_local_LL = ResBlock_conv(64,rkhs_dim,rkhs_dim,init_fn)
        self.psi_local_LG = ResBlock_conv(64,rkhs_dim,rkhs_dim,init_fn)
        self.psi_local_GL = ResBlock_conv(64,rkhs_dim,rkhs_dim,init_fn)

        self.psi_global_LG = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)
        self.psi_global_GL = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)
        self.psi_global_GG = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)

        self.layers = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU(),self.flatten,self.fc_1,nn.ReLU()])
        self.convs  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])

        self.global_encoder = nn.Sequential(*[self.flatten,self.fc_1])
        self.local_encoder  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])


"""
Misc
- ResBlock (Cnn)
- ResBlock (fc)
- Discriminator (fc)
- Noisy layer (W~N(b,sigma^2))
- Flatten (fc)
"""

class Flat_encoder(nn.Module):
  def __init__(self, enc_local,enc_global):
    super().__init__()
    self.enc_local = enc_local
    self.enc_global = enc_global
  def forward(self,x):
    return self.enc_global(self.enc_local(x)).view(x.shape[0],-1)

class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock_2layers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock_2layers, self).__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            Conv2dSame(in_channels, out_channels, 3)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out

class ResBlock_conv(nn.Module):
    """
    Simple 1 hidden layer resblock
    """

    def __init__(self, in_features, hidden_features, out_features, init_fn=lambda m: init(m, nn.init.orthogonal_,lambda x: nn.init.constant_(x, 0))):
        super(ResBlock_conv, self).__init__()

        self.psi_1 = init_fn( nn.Conv2d(in_features,hidden_features, [1,1], 1, bias=True) )
        self.psi_2 = init_fn( nn.Conv2d(hidden_features,out_features, [1,1], 1, bias=True) )

        self.W = init_fn( nn.Conv2d(in_features,hidden_features, [1,1], 1, bias=True) )

    def forward(self,x,action=None):
        if action is not None:
            x = torch.cat([x,action],dim=1)
        residual = self.W(x)
        x = F.relu(self.psi_1(x))
        x = self.psi_2(x) + residual
        return x

class ResBlock_fc(nn.Module):
    """
    Simple 1 hidden layer resblock (for fully-connected inputs)
    """

    def __init__(self, in_features, hidden_features, out_features,init_fn=lambda m: init(m, nn.init.orthogonal_,lambda x: nn.init.constant_(x, 0))):
        super(ResBlock_fc, self).__init__()

        self.psi_1 =  nn.Linear(in_features, hidden_features, bias=True) 
        self.psi_2 =  nn.Linear(hidden_features, out_features, bias=True) 

        self.W = init_fn( nn.Linear(in_features,out_features,bias=False) )

    def forward(self,x,action=None):
        if action is not None:
            x = torch.cat([x,action],dim=1)
        residual = self.W(x)
        x = F.relu(self.psi_1(x))
        x = self.psi_2(x) + residual
        return x

class OneHot_fc(nn.Module):
    """
    Simple 1 hidden layer resblock for a one-hot vector
    """

    def __init__(self, in_features, hidden_features, out_features,init_fn=lambda m: init(m, nn.init.orthogonal_,lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))):
        super().__init__()

        self.in_features = in_features
        self.psi_1 = init_fn( nn.Linear(in_features, hidden_features) )
        self.psi_2 = init_fn( nn.Linear(hidden_features, out_features) )
        self.W = init_fn( nn.Linear(in_features,out_features,bias=True) )

    def forward(self,x):
        x = make_one_hot(x,self.in_features)
        residual = self.W(x)
        x = F.relu(self.psi_1(x))
        x = self.psi_2(x) + residual
        return x

class ResBlock_conv_light(nn.Module):
    """
    Simple resblock (no hidden layer)
    """

    def __init__(self, in_features,out_features):
        super().__init__()

        self.psi_1 = nn.Conv2d(in_features,out_features, [1,1], 1, bias=True)
        self.psi_2 = nn.Conv2d(out_features,out_features, [1,1], 1, bias=True)

        # self.W = nn.Conv2d(in_features,out_features, [1,1], 1, bias=False)

    def forward(self,x,action=None):
        if action is not None:
            x = torch.cat([x,action],dim=1)
        # residual = self.W(x)
        x = F.relu(self.psi_1(x))# + residual
        x = self.psi_2(x)
        return x

class ResBlock_fc_light(nn.Module):
    """
    Simple resblock (no hidden layer, for fully-connected inputs)
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.psi_1 = nn.Linear(in_features, out_features)
        self.psi_2 = nn.Linear(out_features,out_features)

        # self.W = nn.Linear(in_features,out_features,bias=False)

    def forward(self,x,action=None):
        if action is not None:
            x = torch.cat([x,action],dim=1)
        # residual = self.W(x)
        x = F.relu(self.psi_1(x))# + residual
        x = self.psi_2(x)
        return x

class OneHot_fc_light(nn.Module):
    """
    Simple 1 hidden layer resblock for a one-hot vector
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.psi_1 = nn.Linear(in_features, out_features)
        self.psi_2 = nn.Linear(out_features,out_features)
        # self.W = nn.Linear(in_features,out_features,bias=False)

    def forward(self,x):
        x = make_one_hot(x,self.in_features)
        # residual = self.W(x)
        x = F.relu(self.psi_1(x))#+ residual
        x = self.psi_2(x)
        return x

class ResBlock_conv_FILM(nn.Module):
    """
    Simple 1 hidden layer resblock
    """

    def __init__(self, in_features, hidden_features, out_features, action_features,layernorm=True,init_fn=lambda m: init(m, nn.init.orthogonal_,lambda x: nn.init.constant_(x, 0))):
        super(ResBlock_conv_FILM, self).__init__()

        self.psi_1 = init_fn( nn.Conv2d(in_features,hidden_features, [1,1], 1, bias=True) )
        self.psi_2 = init_fn( nn.Conv2d(hidden_features,out_features, [1,1], 1, bias=True) )

        self.film1 = FILM(hidden_features, action_features, layernorm=layernorm)
        self.film2 = FILM(hidden_features, action_features, layernorm=layernorm)
        self.film3 = FILM(out_features, action_features, layernorm=layernorm)

        self.W = init_fn( nn.Conv2d(in_features,hidden_features, [1,1], 1, bias=True) )

    def forward(self,x,action=None):
        residual = self.film1(self.W(x),action)
        x = self.film2(self.psi_1(x),action)
        x = F.relu(x)
        x = self.film3(self.psi_2(x),action)
        x = x + residual
        return x

class ResBlock_fc_FILM(nn.Module):
    """
    Simple 1 hidden layer resblock (for fully-connected inputs)
    """

    def __init__(self, in_features, hidden_features, out_features,action_features,layernorm=True,init_fn=lambda m: init(m, nn.init.orthogonal_,lambda x: nn.init.constant_(x, 0))):
        super(ResBlock_fc_FILM, self).__init__()

        self.psi_1 =  nn.Linear(in_features, hidden_features, bias=True) 
        self.psi_2 =  nn.Linear(hidden_features, out_features, bias=True) 

        self.film1 = FILM(hidden_features, action_features, layernorm=layernorm)
        self.film2 = FILM(out_features, action_features, layernorm=layernorm)

        self.W = init_fn( nn.Linear(in_features,out_features,bias=True) )

    def forward(self,x,action=None):
        residual = self.W(x)
        x = self.film1(self.psi_1(x),action)
        x = F.relu(x)
        x = self.film2(self.psi_2(x),action)
        x = x + residual
        return x

class OneHot(nn.Module):
    """
    Simple one-hot vector
    """

    def __init__(self, in_features):
        super().__init__()

        self.in_features = in_features
        # self.psi_1 = nn.Linear(in_features, out_features)
        # self.W = nn.Linear(in_features,out_features,bias=False)

    def forward(self,x):
        x = make_one_hot(x,self.in_features)
        # residual = self.W(x)
        # x = (self.psi_1(x))#+ residual
        return x

class Discriminator(nn.Module):
    """
    Simple 1 hidden layer discriminator with softmax
    """

    def __init__(self, in_features, out_features):
        super(Discriminator, self).__init__()

        self.fc_1 = nn.Linear(in_features,128)
        self.fc_2 = nn.Linear(128,128)
        self.fc_3 = nn.Linear(128,out_features)

        self.layers = nn.Sequential(*[
            self.fc_1,
            nn.ReLU(),
            self.fc_2,
            nn.ReLU(),
            self.fc_3,
            nn.SoftMax()
        ])

    def forward(s_t,s_t_p_1):
        s = torch.cat([s_t,s_t_p_1],dim=-1)
        x = self.layers(s)
        return x

class NoisyLinear(nn.Module):
    """
    todo: add reference to Kaixhin's Rainbow etc.
    """

    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input_):
        if self.training:
            return F.linear(input_,
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input_, self.weight_mu, self.bias_mu)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class UnFlatten(nn.Module):
    def forward(self, x, size=3456):
        return x.contiguous().view(x.size(0), size, 1, 1)

class Bilinear(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = torch.nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)

class FILM(nn.Module):
    def __init__(self, input_dim, cond_dim, layernorm=True):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.layernorm = nn.LayerNorm(input_dim, elementwise_affine=False) \
            if layernorm else nn.Identity()
        self.conditioning = nn.Linear(cond_dim, input_dim*2)
    def forward(self, input, cond):
        conditioning = self.conditioning(cond)
        gamma = conditioning[..., :self.input_dim]
        beta = conditioning[..., self.input_dim:]
        if len(input.shape) > 2:
            return self.layernorm(input.permute(0,2,3,1)).permute(0,3,1,2)*gamma.unsqueeze(-1).unsqueeze(-1).repeat(1,1,input.shape[2],input.shape[3]) + beta.unsqueeze(-1).unsqueeze(-1).repeat(1,1,input.shape[2],input.shape[3])
        else:
            return self.layernorm(input)*gamma + beta

# class Reshape(nn.Module):
#     def __init__(self,dims):
#         self.dims = dims
#     def forward(self, x):
#         return x.view(x.size()[0], )

"""
Legacy
Copy-pasted from ray.rllib to avoid installing the lib
"""

class TorchModel(nn.Module):
    """Defines an abstract network model for use with RLlib / PyTorch."""

    def __init__(self, obs_space, num_outputs, options):
        """All custom RLlib torch models must support this constructor.

        Arguments:
            obs_space (gym.Space): Input observation space.
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Dictionary of model options.
        """
        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.num_outputs = num_outputs
        self.options = options

    def forward(self, input_dict, hidden_state):
        """Wraps _forward() to unpack flattened Dict and Tuple observations."""
        input_dict["obs"] = input_dict["obs"].float()  # TODO(ekl): avoid cast
        input_dict["obs"] = restore_original_dimensions(
            input_dict["obs"], self.obs_space, tensorlib=torch)
        outputs, features, vf, h = self._forward(input_dict, hidden_state)
        return outputs, features, vf, h

    def state_init(self):
        """Returns a list of initial hidden state tensors, if any."""
        return []

    def _forward(self, input_dict, hidden_state):
        """Forward pass for the model.

        Prefer implementing this instead of forward() directly for proper
        handling of Dict and Tuple observations.

        Arguments:
            input_dict (dict): Dictionary of tensor inputs, commonly
                including "obs", "prev_action", "prev_reward", each of shape
                [BATCH_SIZE, ...].
            hidden_state (list): List of hidden state tensors, each of shape
                [BATCH_SIZE, h_size].

        Returns:
            (outputs, feature_layer, values, state): Tensors of size
                [BATCH_SIZE, num_outputs], [BATCH_SIZE, desired_feature_size],
                [BATCH_SIZE], and [len(hidden_state), BATCH_SIZE, h_size].
        """
        raise NotImplementedError

class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 in_size,
                 out_size,
                 initializer=None,
                 activation_fn=None,
                 bias_init=0):
        super(SlimFC, self).__init__()
        layers = []
        linear = nn.Linear(in_size, out_size)
        if initializer:
            initializer(linear.weight)
        nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        if activation_fn:
            layers.append(activation_fn())
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class SlimConv2d(nn.Module):
    """Simple mock of tf.slim Conv2d"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel,
                 stride,
                 padding,
                 initializer=nn.init.xavier_uniform_,
                 activation_fn=nn.ReLU,
                 bias_init=0):
        super(SlimConv2d, self).__init__()
        layers = []
        if padding:
            layers.append(nn.ZeroPad2d(padding))
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        if initializer:
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)

        layers.append(conv)
        if activation_fn:
            layers.append(activation_fn())
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)