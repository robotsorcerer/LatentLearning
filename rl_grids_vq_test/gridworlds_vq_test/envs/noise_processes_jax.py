import jax.numpy as jnp
from jax.random import PRNGKey
import jax
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.covariance import EmpiricalCovariance
from functools import partial
from jax import lax

from PIL import Image

def select_atoms(val, el):
    return lax.cond(val, lambda x: -1*x, lambda x: x, el)

@partial(jax.jit, static_argnums=(0, 1))
def iteration_jit(J, beta, config, randoms, mask):
    interactions = jnp.roll(config, 1, 0) + jnp.roll(config, 1, 1) + jnp.roll(config, -1, 0) + jnp.roll(config, -1, 1)
    deltaE = 2 * J * config * interactions
    boltzmann = jnp.exp(-beta * deltaE) * mask
    vv = jax.vmap(select_atoms, (0,0),0)
    config = jax.vmap(vv, (1,1),1)( randoms < boltzmann, config)
    return config

class Ising():
    ''' Simulating the Ising model
        Taken from https://stanczakdominik.github.io/posts/parallelizable-numpy-implementation-of-2d-ising-model/ '''

    def __init__(self, N = 13,  J = 1, beta = 1/0.4):
        self.beta = beta
        self.J = J
        self.N = N
        self.rng = PRNGKey(0)

        self.mask = np.ones((N,N))
        self.mask[::2, ::2] = 0
        self.mask[1::2, 1::2] = 0
        self.mask = jnp.array(self.mask)

        self.config = self.reset()

    def reset(self):
        self.rng, key = jax.random.split(self.rng)
        config = jax.random.randint(key, (self.N,self.N), minval=0, maxval=2) * 2 - 1
        return config
    
    def iteration(self, rng, mask):
        rng, key = jax.random.split(rng)
        randoms = jax.random.uniform(key, self.config.shape)

        new_config = iteration_jit(self.J, self.beta, self.config, randoms, mask)
        
        return new_config, rng

    def step(self, n_steps = 1):
        for _ in np.arange(n_steps):
            self.config, self.rng = self.iteration(self.rng, self.mask)
            self.config, self.rng = self.iteration(self.rng, 1-self.mask)
        return self.config

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    ising = Ising(beta=0.01)
    n_steps = 1000
    start = time.time()
    obs = ising.step(n_steps=n_steps)
    print('Total time [%d frames]: %.3f'%(n_steps, time.time()-start))
    
    