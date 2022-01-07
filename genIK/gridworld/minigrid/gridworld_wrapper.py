import random

from collections import deque
from PIL import Image, ImageDraw
from gym_minigrid.minigrid import *
from .exogenous_noise_util import Circle
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from .minigrid_interface import MiniGridInterface


class GridWorldWrapper(MiniGridInterface):

    def __init__(self, env, config):

        self.timestep = -1  # Current time step
        self.horizon = config["horizon"]
        self.env = env

        self.actions = env.actions
        # self.env = RGBImgPartialObsWrapper(self.env)
        # self.env = ImgObsWrapper(self.env)
        self.circles = None

        self.height, self.width = config["height"] * config["tile_size"], config["width"] * config["tile_size"]

        self.colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(8)]
        self.circle_width = config["circle_width"]
        self.circle_motion = config["circle_motion"]
        self.num_circles = config["num_circles"]

        # with open("%s/progress.csv" % config["save_path"], "w") as f:
        #     f.write("Episode,     Moving Avg.,     Mean Return\n")
        # self.moving_avg = deque([], maxlen=10)
        # self.sum_return = 0
        # self.num_eps = 0
        # self._eps_return = 0.0
        # self.save_path = config["save_path"]
        self.feature_type = config["feature_type"]
        self.tile_size = config["tile_size"]

        ####
        self.img_ctr = 0
        ####


    def get_pos(self):
        state = np.copy(self.env.agent_pos)
        return state 

    def get_endogenous_state(self, state):
        return self.get_state

    def reset(self, generate_obs=True, agent_pos=None):
        """
            :return:
                obs:        Agent observation. No assumption made on the structure of observation.
                info:       Dictionary containing relevant information such as latent state, etc.
        """

        self.env.reset()
        if agent_pos is not None:
            self.env.agent_pos = agent_pos
        self.timestep = 0
        state = (self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir, self.timestep)
        info = {
            "state": state,
            "endogenous_state": state,
            "time_step": self.timestep
        }

        # # Generate circles
        self.circles = []
        for _ in range(0, self.num_circles):
            # Generate random four points
            coord = [random.randint(0, self.width // 2), random.randint(0, self.height // 2),
                     random.randint(self.width // 2, self.width), random.randint(self.height // 2, self.height)]
            color = random.choice(self.colors)
            self.circles.append(Circle(coord, color, self.circle_width))

        if generate_obs:
            img = self.env.render('rgb_array', tile_size=self.tile_size, highlight=False)
            img = self.generate_image(img)
        else:
            img = None

        return img, info
    
    def reset_goal(self, goal_pos=None):
        raise NotImplementedError

    def generate_image(self, img):

        width, height = img.shape[0], img.shape[1]
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)

        for circle in self.circles:
            draw.ellipse(circle.coord, outline=circle.color, width=circle.width)
        exo_im = np.array(image).astype(np.uint8)

        img_shape = img.shape
        exo_im = exo_im.reshape((-1, 3))
        img = img.reshape((-1, 3))
        obs_max = img.max(1)
        bg_pixel_ix = np.argwhere(obs_max < 100)  # flattened (x, y) position where pixels are black in color
        values = np.squeeze(exo_im[bg_pixel_ix])
        np.put_along_axis(img, bg_pixel_ix, values, axis=0)
        img = img.reshape(img_shape)

        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        self.img_ctr += 1
        plt.savefig("./visual_gridworld_%d.pdf" % self.img_ctr, bbox_inches='tight')
        # if self.img_ctr == 2:
        #     exit(0)

        img = img / 255.0
        # img = color.rgb2gray(img / 255.0)

        return img

    def perturb(self):
        self.circles = [self.perturb_circle(circle, self.height, self.width) for circle in self.circles]

    def perturb_circle(self, circle, height, width):

        # Each of the four coordinate is moved independently by 10% of the corresponding dimension
        r = [random.choice([-1, 1]) for _ in range(4)]
        coord = circle.coord[0] + r[0] * int(self.circle_motion * width), \
                circle.coord[1] + r[1] * int(self.circle_motion * height), \
                circle.coord[2] + r[2] * int(self.circle_motion * width), \
                circle.coord[3] + r[3] * int(self.circle_motion * height)

        return Circle(coord=coord, color=circle.color, width=circle.width)

    def step(self, action, generate_obs=True):
        """
            :param action:
            :param generate_obs: If True then observation is generated, otherwise, None is returned
            :return:
                obs:        Agent observation. No assumption made on the structure of observation.
                reward:     Reward received by the agent. No Markov assumption is made.
                done:       True if the episode has terminated and False otherwise.
                info:       Dictionary containing relevant information such as latent state, etc.
        """

        if self.timestep > self.horizon:
            raise AssertionError("Cannot take more actions than horizon %d" % self.horizon)

        obs, reward, done, info = self.env.step(action)
        self.timestep += 1

        # self._eps_return += reward

        done = done or self.timestep == self.horizon

        state = (self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir, self.timestep)
        info = {
            "state": state,
            "endogenous_state": state,
            "time_step": self.timestep
        }

        self.perturb()

        if generate_obs:
            img = self.env.render('rgb_array', tile_size=self.tile_size, highlight=False)
            img = self.generate_image(img)
        else:
            img = None

        return img, reward, done, info

    def save(self, save_path, fname=None):
        """
            Save the environment
            :param save_path:   Save directory
            :param fname:       Additionally, a file name can be provided. If save is a single file, then this will be
                                used else it can be ignored.
            :return: None
        """
        pass

    def load(self, load_path, fname=None):
        """
            Save the environment
            :param load_path:   Load directory
            :param fname:       Additionally, a file name can be provided. If load is a single file, then only file
                                with the given fname will be used.
            :return: Environment
        """
        raise NotImplementedError()

    def is_episodic(self):
        """
            :return:                Return True or False, True if the environment is episodic and False otherwise.
        """
        return True