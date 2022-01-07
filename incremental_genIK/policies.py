import torch
import torch.nn as nn
import torch.nn.functional as F




class StationaryStochasticPolicy(nn.Module):

    def __init__(self, num_actions, obs_dim):
        super(StationaryStochasticPolicy, self).__init__()

        self.layer1 = nn.Linear(obs_dim, 56)
        self.layer2 = nn.Linear(56, 56)
        self.layer3 = nn.Linear(56, num_actions)

    def gen_prob(self, observations):
        observations = (torch.from_numpy(observations)).float()
        x = F.relu(self.layer1(observations))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=0)

        return x

    def sample_action(self, observations):
        prob = self.gen_prob(observations)
        dist = torch.distributions.Categorical(prob)

        action = (torch.multinomial(dist.probs, 1, True)).item()

        return action

    def get_argmax_action(self, observations):
        prob = self.gen_prob(observations)
        action = (prob.max(0)[1]).item()

        return action