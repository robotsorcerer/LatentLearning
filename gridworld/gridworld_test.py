import random
import matplotlib.pyplot as plt

from gridworld.gridworld_wrapper import GridWorldWrapper

env = GridWorldWrapper.make_env("twogrids")

plt.ion()
obs, info = env.reset()
plt.imshow(obs)
plt.show()
plt.pause(0.5)

for h in range(0, 100):
    action = random.randint(0, 4)
    obs, reward, done, info = env.step(action)
    plt.clf()
    plt.imshow(obs)
    plt.show()
    plt.pause(0.5)
