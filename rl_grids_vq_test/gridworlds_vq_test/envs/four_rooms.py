import numpy as np
from gym import core, spaces
from gym.envs.registration import register
import gym

class FourRooms:
    def __init__(self,goal,viz_params=[]):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""     
        self.viz_params = viz_params
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
                                    low=0,
                                    high=255,
                                    shape=(13,13,3),
                                    dtype=np.uint8,
                                )
        self.state_space = spaces.Discrete(np.sum(self.occupancy == 0))
        self.reward_range = np.array([0,1])
        self.metadata = {}

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)
        self._seed = self.rng

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = self.tostate[goal]
        self.init_states = list(range(self.state_space.n))
        self.init_states.remove(self.goal)

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self,state=None):
        if state is None:
            state = self.tocell[np.random.choice(self.init_states,size=1)[0]]
        self.currentcell = state
        if 'pixel' in self.viz_params:
            return self.to_pixel(state)
        return state

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]
        done = state == self.goal
        if 'pixel' in self.viz_params:
            state = self.to_pixel(self.currentcell)
        return state, float(done), done, None

    def to_pixel(self, state):
        obs = np.stack([self.occupancy]*3).transpose(1,2,0)
        obs[obs==1] = 192
        obs[obs==0] = 0
        goal = self.tocell[self.goal]
        obs[goal[0],goal[1]] = [24,134,45]
        obs[state[0],state[1]] = [170,1,20]
        return obs

if __name__ == "__main__":
    import gym
    env = FourRooms(goal=(10,10),viz_params=['pixel'])
    import matplotlib.pyplot as plt
    state = env.reset((1,1))
    state, _,_, _ = env.step(3)