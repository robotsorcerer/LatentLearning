import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
from PIL import Image, ImageDraw

from . import grid
from .objects.agent import Agent
from .objects.depot import Depot

from .exo_noise_circle import Circle
#from .sensors import *


class GridWorld(grid.BaseGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = Agent()
        self.actions = [i for i in range(4)]
        self.action_map = grid.directions
        self.agent.position = np.asarray((0, 0), dtype=int)
        self.goal = None
        self.exo_noise_set = False

        self.config = {
                  "num_circles": 4,
                  "circle_width": 8,
                  "circle_motion": 0.05 ### default : 0.05
        }

        self.width = self._cols
        self.height = self._rows

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

        s  = self.get_state()

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
        state = np.copy(self.agent.position)
        return state 

    def plot(self, ax=None):
        ax = super().plot(ax)
        if self.agent:
            self.agent.plot(ax)
        if self.goal:
            self.goal.plot(ax)
        return ax

    def get_obs(self, ax=None, with_exo_noise=False):
        if ax is None:
            ax = self.plot()
        # convert to np of 80x80
        ax.figure.set_dpi(80/max(self._rows,self._cols)) # add scale ?
        ax.figure.tight_layout(pad=0)
        canvas = FigureCanvas(ax.figure)
        canvas.draw()
        buf = canvas.tostring_rgb()
        w, h = canvas.get_width_height()
        X = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
        plt.close('all')

        # add noise
        if with_exo_noise:
            if self.exo_noise_set:
                X = self.generate_noised_image(X)
            else:
                warnings.warn("Asked for exo_noise, but exo_noise not configured ! Returning obs without noise ...")
        return X
    
    def generate_noised_image(self, img):
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

    def set_exo_noise_config(self, config):
        self.exo_noise_set = True
        self.circles = None
        self.colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(8)]
        self.circle_width = config["circle_width"]
        self.circle_motion = config["circle_motion"]
        self.num_circles = config["num_circles"]
        ## Generate circles
        # width, height = self._cols, self._rows
        self.scale_circ = 80/max(self._cols, self._rows)
        self.sample_circles(self._rows, self._cols, self.scale_circ)

    def sample_circles(self, width, height, scale):
        self.circles = []
        for _ in range(0, self.num_circles):
            # Generate random four points
            coord = [random.randint(0, width // 2)*scale, random.randint(0, height // 2)*scale,
                     random.randint(width // 2, width)*scale, random.randint(height // 2, height)*scale]
            color = random.choice(self.colors)
            self.circles.append(Circle(coord, color, self.circle_width))

    def perturb_circles(self):
        self.circles = [self.perturb_circle(circle, self._cols, self._rows) for circle in self.circles]

    def perturb_circle(self, circle, width, height):
        # Each of the four coordinate is moved independently by 10% of the corresponding dimension
        # circle_motion = corr_noise
        if corr_noise > 0.0 : 
            movement = 2
        else:
            movement = 1
        r = [random.choice([-movement, movement]) for _ in range(4)]
        coord = circle.coord[0] + r[0] * int(self.circle_motion * width), \
                circle.coord[1] + r[1] * int(self.circle_motion * height), \
                circle.coord[2] + r[2] * int(self.circle_motion * width), \
                circle.coord[3] + r[3] * int(self.circle_motion * height)

        return Circle(coord=coord, color=circle.color, width=circle.width)

#####################################################################
#####################################################################

    def get_image(self, obs, exo_noise, corr_noise):
        self.set_exo_noise_config(self.config)

        if corr_noise > 0.0 : 
            print ("RGB Images under Correlated Exo Noise")
            images = np.zeros([obs.shape[0], obs.shape[1], obs.shape[1] ])
            for k in range(obs.shape[0]):
                im_perturb = (self.generate_image(obs[k, :, :], exo_noise, corr_noise)).reshape(1, obs.shape[1], obs.shape[1])
                images[k, :, :] = im_perturb
                # im = self.plot(corr_noise)
                # plt.imshow(im)
                # plt.savefig("./exo_noise_gridworld%d.pdf" % k, bbox_inches='tight')
        else : 
            images = self.generate_image(obs, exo_noise, corr_noise)
        return images        

    def generate_image(self, img, exo_noise, corr_noise):
        width, height = img.shape[0], img.shape[1]
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)

        if exo_noise:
            for circle in self.circles:
                self.perturb(corr_noise)
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
        img = img / 255.0


        return img

    def sensor_observation(self, raw_state):
        sensor_list = []
        sensor_list.append(RearrangeXYPositionsSensor((self._rows, self._cols)))

        sensor_list += [
            OffsetSensor(offset=(0.5, 0.5)),
            NoisySensor(sigma=0.05),
            ImageSensor(range=((0, self._rows), (0, self._cols)), pixel_density=3),
            # ResampleSensor(scale=2.0),
            BlurSensor(sigma=0.6, truncate=1.),
            NoisySensor(sigma=0.01)
        ]
        sensor = SensorChain(sensor_list)
        X = sensor.observe(raw_state)
        return X

    def get_image(self, obs, exo_noise):
        img = self.generate_image(obs)
        return img        

    def get_image(self, obs, exo_noise, corr_noise):
        self.set_exo_noise_config(self.config)

        if corr_noise > 0.0 : 
            print ("RGB Images under Correlated Exo Noise")
            images = np.zeros([obs.shape[0], obs.shape[1], obs.shape[1] ])
            for k in range(obs.shape[0]):
                im_perturb = (self.generate_image(obs[k, :, :], exo_noise, corr_noise)).reshape(1, obs.shape[1], obs.shape[1])
                images[k, :, :] = im_perturb
                # im = self.plot(corr_noise)
                # plt.imshow(im)
                # plt.savefig("./exo_noise_gridworld%d.pdf" % k, bbox_inches='tight')
        else : 
            images = self.generate_image(obs, exo_noise, corr_noise)
        return images
    
    def sensor_observation(self, raw_state):
        sensor_list = []
        sensor_list.append(RearrangeXYPositionsSensor((self._rows, self._cols)))

        sensor_list += [
            OffsetSensor(offset=(0.5, 0.5)),
            NoisySensor(sigma=0.05),
            ImageSensor(range=((0, self._rows), (0, self._cols)), pixel_density=3),
            # ResampleSensor(scale=2.0),
            BlurSensor(sigma=0.6, truncate=1.),
            NoisySensor(sigma=0.01)
        ]
        sensor = SensorChain(sensor_list)
        X = sensor.observe(raw_state)
        return X       

    def perturb(self, corr_noise):
        self.circles = [self.perturb_circle(circle, self.height, self.width, corr_noise) for circle in self.circles]

    def perturb_circle(self, circle, height, width, corr_noise):
        # Each of the four coordinate is moved independently by 10% of the corresponding dimension
        self.circle_motion = corr_noise
        if corr_noise > 0.0 : 
            movement = 2
        else:
            movement = 1
        r = [random.choice([-movement, movement]) for _ in range(4)]
        coord = circle.coord[0] + r[0] * int(self.circle_motion * width), \
                circle.coord[1] + r[1] * int(self.circle_motion * height), \
                circle.coord[2] + r[2] * int(self.circle_motion * width), \
                circle.coord[3] + r[3] * int(self.circle_motion * height)

        return Circle(coord=coord, color=circle.color, width=circle.width)


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








