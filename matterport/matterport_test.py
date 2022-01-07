import random
import matplotlib.pyplot as plt

from matterport.matterport import Matterport

env = Matterport.make()

plt.ion()
obs, info = env.reset()
plt.imshow(obs)
plt.show()
plt.pause(0.5)

for h in range(0, 100):
    action = random.choice(env.actions)
    obs, reward, done, info = env.step(action)
    plt.clf()
    plt.imshow(obs)
    plt.show()
    plt.pause(0.5)
