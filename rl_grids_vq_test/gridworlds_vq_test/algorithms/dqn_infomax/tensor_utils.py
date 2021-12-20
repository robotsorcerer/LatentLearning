import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from noisy_state_abstractions.algorithms.dqn_infomax.utils import rollout

class Model(nn.Module):
    def __init__(self,top,bottom):
        nn.Module.__init__(self)
        self.features = bottom
        self.classifier = top
    
    def forward(self,inputs):

        _, outputs_bottom, _, _ = self.features({"obs": inputs}, None)
        outputs_top, _, _, _ = self.classifier({"obs":outputs_bottom}, None)
        
        return outputs_top, outputs_bottom

def shuffle_joint(x):
    """
    Shuffles input x on first (batch) dimension
    Args:
        x (torch.Tensor): (n,**) A >1D tensor, gets shuffled along batch dimension
    """
    n = len(x)
    idx = np.array(range(n))
    np.random.shuffle(idx)
    return x[idx]

def random_locs_2d(x, k_hot=1):
    '''
    Sample a k-hot mask over spatial locations for each set of conv features
    in x, where x.shape is like (n_batch, n_feat, n_x, n_y).
        W
    ############
    # x   x  x #
    #x   x   xx#  H
    #  x   x x #
    ############

    (Taken directly from AMDIM)
    Args:
        x (torch.FloatTensor): (batch,N_channels,H,W) Input tensor of RKHS scores, out of which we pick k_hot locations
        k_hot (int): Number of sample per (H,W) slice
    '''
    # assume x is (n_batch, n_feat, n_x, n_y)
    x_size = x.size()
    n_batch = x_size[0]
    n_locs = x_size[2] * x_size[3]
    idx_topk = torch.topk(torch.rand((n_batch, n_locs)), k=k_hot, dim=1)[1]
    khot_mask = torch.zeros((n_batch, n_locs)).scatter_(1, idx_topk, 1.)
    rand_locs = khot_mask.reshape((n_batch, 1, x_size[2], x_size[3]))
    # rand_locs = maybe_half(rand_locs)
    return rand_locs

def wide_to_long_idx(shape,idx):
    """
    Transforms wide index (-1,4) into long index (-1)

    Args:
        shape (list): Shape of the score tensor indexed by idx. Assumed 4-dimensional (batch,N_channels,H,W)
        idx (torch.LongTensor): Index tensor of dimension (-1,4)
    """
    return idx[:,0] * (shape[1] * shape[2] * shape[3]) + idx[:,1] * (shape[2] * shape[3]) + idx[:,2] * (shape[3] ) + idx[:,3] 

def get_fibers_from_increments(reference_idx,increments,encoder_scores,clip_or_wrap):
    """"
    Selects new fibers corresponding to increments passed as input
    Args:
        reference_idx (torch.LongTensor): (batch,N_reference,4) Indices of reference fibers
        increments (torch.LongTensor): (batch,N_reference,4) Increments either <r or >r
        encoder_scores (torch.FloatTensor): (batch,N_channels,H,W) RKHS-encoded scores (output of encoder convnet)
        clip_or_wrap (str): Whether to 'clip' out-of-box indices (positive samples), or 'wrap' them around the tensor (negative samples)
    """

    encoder_shape = encoder_scores.shape
    idx = reference_idx
    
    if clip_or_wrap == 'clip':
        idx[:,:,2:] = torch.clamp(idx[:,:,2:] + increments[:,:,2:],0,encoder_shape[-1]-1)
    elif clip_or_wrap == 'wrap':
        idx[:,:,2:] = (idx[:,:,2:] + increments[:,:,2:]) % encoder_shape[-1]
    N_reference = idx.shape[1]
    idx = idx.view(-1,4)
    N_channels = encoder_shape[1]

    idx_fibers = idx.repeat([1,N_channels]).view(-1,4)
    idx_fibers[:,1] = torch.arange(0,N_channels).repeat(encoder_shape[0] * N_reference)
    
    idx_to_keep = wide_to_long_idx(encoder_shape,idx_fibers)

    samples = encoder_scores.view(-1)[idx_to_keep].view(-1,N_reference,N_channels).permute(0,2,1)

    return samples

def find_element(tensor,dim_1, el):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    shape = tensor.shape
    return ((tensor==el).nonzero().to(device)).view(shape[0],dim_1,-1)

def tanh_clip(x, clip_val=20.):
    '''
    soft clip values to the range [-clip_val, +clip_val]
    '''
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip