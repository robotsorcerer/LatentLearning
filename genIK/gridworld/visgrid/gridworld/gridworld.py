import random

import numpy as np
import matplotlib.pyplot as plt

from . import grid
from .objects.agent import Agent
from .objects.depot import Depot

class GridWorld(grid.BaseGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = Agent()
        self.actions = [i for i in range(4)]
        self.action_map = grid.directions
        self.agent.position = np.asarray((0, 0), dtype=int)
        self.goal = None

    def reset_agent(self, pos=None):
        if pos:
            self.agent.position = pos
        else:
            self.agent.position = self.get_random_position()
        at = lambda x, y: np.all(x.position == y.position)
        while (self.goal is not None) and at(self.agent, self.goal):
            self.agent.position = self.get_random_position()

    def reset_goal(self, goal_pos=None):
        if self.goal is None:
            self.goal = Depot()
        if goal_pos:
            self.goal.position = goal_pos
        else:
            self.goal.position = self.get_random_position()
        self.reset_agent()

    def step(self, action):
        assert (action in range(4))
        direction = self.action_map[action]
        if not self.has_wall(self.agent.position, direction):
            self.agent.position += direction
        s = self.get_state()
        if self.goal:
            at_goal = np.all(self.agent.position == self.goal.position)
            r = 0 if at_goal else -1
            done = True if at_goal else False
        else:
            r = 0
            done = False
        return s, r, done

    def can_run(self, action):
        assert (action in range(4))
        direction = self.action_map[action]
        return False if self.has_wall(self.agent.position, direction) else True

    def get_state(self):
        return np.copy(self.agent.position)

    def plot(self, ax=None):
        ax = super().plot(ax)
        if self.agent:
            self.agent.plot(ax)
        if self.goal:
            self.goal.plot(ax)
        return ax

class TestWorld(GridWorld):
    def __init__(self):
        super().__init__(rows=3, cols=4)
        self._grid[1, 4] = 1
        self._grid[2, 3] = 1
        self._grid[3, 2] = 1
        self._grid[5, 4] = 1
        self._grid[4, 7] = 1

        # Should look roughly like this:
        # _______
        #|  _|   |
        #| |    _|
        #|___|___|

class RingWorld(GridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for r in range(self._rows - 2):
            self._grid[2 * r + 3, 2] = 1
            self._grid[2 * r + 3, 2 * self._cols - 2] = 1
        for c in range(self._cols - 2):
            self._grid[2, 2 * c + 3] = 1
            self._grid[2 * self._rows - 2, 2 * c + 3] = 1

class SnakeWorld(GridWorld):
    def __init__(self):
        super().__init__(rows=3, cols=4)
        self._grid[1, 4] = 1
        self._grid[2, 3] = 1
        self._grid[2, 5] = 1
        self._grid[3, 2] = 1
        self._grid[3, 6] = 1
        self._grid[5, 4] = 1

        # Should look roughly like this:
        # _______
        #|  _|_  |
        #| |   | |
        #|___|___|

class MazeWorld(GridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        walls = []
        for row in range(0, self._rows):
            for col in range(0, self._cols):
                #add vertical walls
                self._grid[row * 2 + 2, col * 2 + 1] = 1
                walls.append((row * 2 + 2, col * 2 + 1))

                #add horizontal walls
                self._grid[row * 2 + 1, col * 2 + 2] = 1
                walls.append((row * 2 + 1, col * 2 + 2))

        random.shuffle(walls)

        cells = []
        #add each cell as a set_text
        for row in range(0, self._rows):
            for col in range(0, self._cols):
                cells.append({(row * 2 + 1, col * 2 + 1)})

        #Randomized Kruskal's Algorithm
        for wall in walls:
            if (wall[0] % 2 == 0):

                def neighbor(set):
                    for x in set:
                        if (x[0] == wall[0] + 1 and x[1] == wall[1]):
                            return True
                        if (x[0] == wall[0] - 1 and x[1] == wall[1]):
                            return True
                    return False

                neighbors = list(filter(neighbor, cells))
                if (len(neighbors) == 1):
                    continue
                cellSet = neighbors[0].union(neighbors[1])
                cells.remove(neighbors[0])
                cells.remove(neighbors[1])
                cells.append(cellSet)
                self._grid[wall[0], wall[1]] = 0
            else:

                def neighbor(set):
                    for x in set:
                        if (x[0] == wall[0] and x[1] == wall[1] + 1):
                            return True
                        if (x[0] == wall[0] and x[1] == wall[1] - 1):
                            return True
                    return False

                neighbors = list(filter(neighbor, cells))
                if (len(neighbors) == 1):
                    continue
                cellSet = neighbors[0].union(neighbors[1])
                cells.remove(neighbors[0])
                cells.remove(neighbors[1])
                cells.append(cellSet)
                self._grid[wall[0], wall[1]] = 0

    @classmethod
    def load_maze(cls, rows, cols, seed):
        env = GridWorld(rows=rows, cols=cols)
        maze_file = 'visgrid/gridworld/mazes/mazes_{rows}x{cols}/seed-{seed:03d}/maze-{seed}.txt'.format(
            rows=rows, cols=cols, seed=seed)
        try:
            env.load(maze_file)
        except IOError as e:
            print()
            print('Could not find standardized {rows}x{cols} maze file for seed {seed}. Maybe it needs to be generated?'.format(rows=rows, cols=cols, seed=seed))
            print()
            raise e
        return env

class SpiralWorld(GridWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add all walls
        for row in range(0, self._rows):
            for col in range(0, self._cols):
                #add vertical walls
                self._grid[row * 2 + 2, col * 2 + 1] = 1

                #add horizontal walls
                self._grid[row * 2 + 1, col * 2 + 2] = 1

        # Check dimensions to decide on appropriate spiral direction
        if self._cols > self._rows:
            direction = 'cw'
        else:
            direction = 'ccw'

        # Remove walls to build spiral
        for i in range(0, min(self._rows, self._cols)):
            # Create concentric hooks, and connect them after the first to build spiral
            if direction == 'ccw':
                self._grid[(2 * i + 1):-(2 * i + 1), (2 * i + 1)] = 0
                self._grid[-(2 * i + 2), (2 * i + 1):-(2 * i + 1)] = 0
                self._grid[(2 * i + 1):-(2 * i + 1), -(2 * i + 2)] = 0
                self._grid[(2 * i + 1), (2 * i + 3):-(2 * i + 1)] = 0
                if i > 0:
                    self._grid[2 * i, 2 * i + 1] = 0

            else:
                self._grid[(2 * i + 1), (2 * i + 1):-(2 * i + 1)] = 0
                self._grid[(2 * i + 1):-(2 * i + 1), -(2 * i + 2)] = 0
                self._grid[-(2 * i + 2), (2 * i + 1):-(2 * i + 1)] = 0
                self._grid[(2 * i + 3):-(2 * i + 1), (2 * i + 1)] = 0
                if i > 0:
                    self._grid[2 * i + 1, 2 * i] = 0

class LoopWorld(SpiralWorld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check dimensions to decide on appropriate spiral direction
        if self._cols > self._rows:
            direction = 'cw'
        else:
            direction = 'ccw'

        if direction == 'ccw':
            self._grid[-3, -4] = 0
        else:
            self._grid[-4, -3] = 0


if __name__ == '__main__':
    grid = SpiralWorld(6, 6)
    grid.reset_goal([2, 3])
    grid.plot()
    plt.show()