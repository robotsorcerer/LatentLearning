import random
import matplotlib.pyplot as plt

from gridworld.gridworld_wrapper import GridWorldWrapper

env = GridWorldWrapper.make_env("gridworld1")

plt.ion()
obs, info = env.reset()

for h in range(0, 100):
    action = random.randint(0, 4)
    obs, reward, done, info = env.step(action)
    plt.clf()
    plt.imshow(obs)
    plt.show()
    plt.pause(0.5)
