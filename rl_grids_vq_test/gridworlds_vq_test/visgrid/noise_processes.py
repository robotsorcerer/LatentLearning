import numpy as np

def iteration(J, beta, config, randoms, mask):
    interactions = np.roll(config, 1, 0) + np.roll(config, 1, 1) + np.roll(config, -1, 0) + np.roll(config, -1, 1)
    deltaE = 2 * J * config * interactions
    boltzmann = np.exp(-beta * deltaE) * mask
    flip_these = randoms<boltzmann
    config[flip_these] *= -1
    return config

class Ising():
    ''' Simulating the Ising model
        Taken from https://stanczakdominik.github.io/posts/parallelizable-numpy-implementation-of-2d-ising-model/ '''

    def __init__(self, N = 13,  J = 1, beta = 1/0.4):
        self.beta = beta
        self.J = J
        self.N = N

        self.mask = np.ones((N, N))
        self.mask[::2, ::2] = 0
        self.mask[1::2, 1::2] = 0
        self.mask = np.array(self.mask)

        self.config = self.reset()

    def reset(self):
        config = np.random.randint(low=0, high=2, size=(self.N,self.N)) * 2 - 1
        self.step = 0
        return config
    
    def iteration(self, mask):
        randoms = np.random.uniform(size=self.config.shape)
        new_config = iteration(self.J, self.beta, self.config, randoms, mask)
        self.step += 1
        return new_config

    def step(self, n_steps = 1):
        for _ in np.arange(n_steps):
            self.config = self.iteration( self.mask)
            self.config = self.iteration( 1-self.mask)
        return self.config

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    ising = Ising(beta=0.01)
    n_steps = 1000
    start = time.time()
    obs = ising.step(n_steps=n_steps)
    print('Total time [%d frames]: %.3f'%(n_steps, time.time()-start))
    
    