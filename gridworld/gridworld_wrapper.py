import json
import random

from collections import deque
from PIL import Image, ImageDraw
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from gridworld.gridworld1 import GridWorld1
from gridworld.gridworld2 import GridWorld2


class GridWorldWrapper:

    def __init__(self, env, config):

        self.timestep = -1  # Current time step
        self.horizon = config["horizon"]
        self.actions = config["actions"]
        self.env = env

        # self.env = RGBImgPartialObsWrapper(self.env)
        # self.env = ImgObsWrapper(self.env)
        self.height, self.width = config["height"] * config["tile_size"], config["width"] * config["tile_size"]

        with open("%s/progress.csv" % config["save_path"], "w") as f:
            f.write("Episode,     Moving Avg.,     Mean Return\n")

        self.moving_avg = deque([], maxlen=10)
        self.sum_return = 0
        self.num_eps = 0
        self._eps_return = 0.0
        self.save_path = config["save_path"]
        self.feature_type = config["feature_type"]
        self.tile_size = config["tile_size"]

        ####
        self.img_ctr = 0
        ####

    def act_to_str(self, action):

        if action == 0:
            return "left"
        elif action == 1:
            return "right"
        elif action == 2:
            return "forward"
        elif action == 3:
            return "left+forward"
        elif action == 4:
            return "right+forward"
        else:
            raise AssertionError("Action must be in {0, 1, 2, 3, 4}")

    def start(self):
        raise NotImplementedError()

    def transition(self, state, action):
        raise NotImplementedError()

    def reward(self, state, action, new_state):
        raise NotImplementedError()

    def get_env_name(self):
        raise NotImplementedError()

    def get_actions(self):
        raise NotImplementedError()

    def get_num_actions(self):
        raise NotImplementedError()

    def get_horizon(self):
        raise NotImplementedError()

    def get_endogenous_state(self, state):
        return state

    def reset(self, generate_obs=True):
        """
            :return:
                obs:        Agent observation. No assumption made on the structure of observation.
                info:       Dictionary containing relevant information such as latent state, etc.
        """

        self.env.reset()
        self.timestep = 0
        state = (self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir, self.timestep)
        info = {
            "state": state,
            "timestep": self.timestep
        }

        if self.num_eps > 0:

            self.moving_avg.append(self._eps_return)
            self.sum_return += self._eps_return

            if self.num_eps % 100 == 0:
                mov_avg = sum(self.moving_avg) / float(len(self.moving_avg))
                mean_result = self.sum_return / float(self.num_eps)

                with open("%s/progress.csv" % self.save_path, "a") as f:
                    f.write("%d,     %f,    %f\n" % (self.num_eps, mov_avg, mean_result))

        self._eps_return = 0.0
        self.num_eps += 1  # Index of current episode starting from 0

        if generate_obs:
            img = self.env.render('rgb_array', tile_size=self.tile_size, highlight=False)
            img = self.generate_image(img)
        else:
            img = None

        return img, info

    def generate_image(self, img):

        width, height = img.shape[0], img.shape[1]
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        exo_im = np.array(image).astype(np.uint8)

        img_shape = img.shape
        exo_im = exo_im.reshape((-1, 3))
        img = img.reshape((-1, 3))
        obs_max = img.max(1)
        bg_pixel_ix = np.argwhere(obs_max < 100)  # flattened (x, y) position where pixels are black in color
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

        self._eps_return += reward

        done = done or self.timestep == self.horizon

        state = (self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir, self.timestep)
        info = {
            "state": state,
            "timestep": self.timestep
        }

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

    def generate_homing_policy_validation_fn(self):
        """
            :return:                Returns a validation function to test for exploration success
        """
        return None

    @staticmethod
    def adapt_config(config):
        """
            Adapt configuration file based on the environment
        :return:
        """
        raise NotImplementedError()

    def num_completed_episode(self):
        """
            :return:    Number of completed episode
        """

        return max(0, self.num_eps - 1)

    def get_mean_return(self):
        """
            :return:    Get mean return of the agent
        """
        return self.sum_return / float(max(1, self.num_completed_episode()))

    def get_optimal_value(self):
        """
            Return V* value
        :return:
        """

        # 1 for reaching goal in the fasted way and paying -0.01 for every step except the last
        # Note that once we reach the goal, the agent gets stuck and only gets reward of 0
        return 1.0 + (self.env.min_dist_to_goal - 1) * -0.01

    @staticmethod
    def make_env(env_name):

        with open("./discrete-factors/configs/%s.json" % env_name, "r") as f:
            config = json.load(f)

        if env_name == "gridworld1":
            base_env = GridWorld1(config)

        elif env_name == "gridworld2":
            base_env = GridWorld2(config)

        else:
            raise NotImplementedError("Environment %s not found" % env_name)

        env = GridWorldWrapper(base_env, config)
        return env
