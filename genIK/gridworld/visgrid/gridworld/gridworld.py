import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
from PIL import Image, ImageDraw

from . import grid
from .objects.agent import Agent
from .objects.depot import Depot

from .exo_noise_circle import Circle

class GridWorld(grid.BaseGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = Agent()
        self.actions = [i for i in range(4)]
        self.action_map = grid.directions
        self.agent.position = np.asarray((0, 0), dtype=int)
        self.goal = None
        self.exo_noise = False

    def set_exo_noise_config(self, config):
        self.exo_noise = True
        self.circles = None
        self.colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(8)]
        self.circle_width = config["circle_width"]
        self.circle_motion = config["circle_motion"]
        self.num_circles = config["num_circles"]
        # # Generate circles
        self.circles = []
        width, height = self._cols, self._rows
        scale_circ = 80/max(width,height)
        for _ in range(0, self.num_circles):
            # Generate random four points
            coord = [random.randint(0, width // 2)*scale_circ, random.randint(0, height // 2)*scale_circ,
                     random.randint(width // 2, width)*scale_circ, random.randint(height // 2, height)*scale_circ]
            color = random.choice(self.colors)
            self.circles.append(Circle(coord, color, self.circle_width))

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
        if self.exo_noise:
            ax.figure.set_dpi(80/max(self._rows,self._cols)) # add scale ?
            ax.figure.tight_layout(pad=0)
            canvas = FigureCanvas(ax.figure)
            canvas.draw()
            buf = canvas.tostring_rgb()
            w, h = canvas.get_width_height()
            X = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
            img = self.generate_image(X)
            return img
        return ax
    
    def generate_image(self, img):

        width, height = img.shape[0], img.shape[1]
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)

        for circle in self.circles:
            draw.ellipse(circle.coord, outline=circle.color, width=circle.width)
        exo_im = np.array(image).astype(np.uint8)
        # make bgk white
        exo_im = np.where(exo_im==0, 255, exo_im)

        img_shape = img.shape
        exo_im = exo_im.reshape((-1, 3))
        img = img.reshape((-1, 3))
        obs_max = img.max(1)
        bg_pixel_ix = np.argwhere(obs_max > 250)  # flattened (x, y) position where pixels are white in color
        values = np.squeeze(exo_im[bg_pixel_ix])
        np.put_along_axis(img, bg_pixel_ix, values, axis=0)
        img = img.reshape(img_shape)

        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.axis('off')
        # plt.tight_layout()
        # self.img_ctr += 1
        # plt.savefig("./visual_gridworld_%d.pdf" % self.img_ctr, bbox_inches='tight')
        # if self.img_ctr == 2:
        #     exit(0)

        img = img / 255.0
        # img = color.rgb2gray(img / 255.0)

        return img

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