import random

from gym_minigrid.minigrid import *


class TwoGrids(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    env_name = "twogrids"

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        left_forward = 3
        right_forward = 4

    def __init__(self, config):

        width = config["width"]
        height = config["height"]
        horizon = config["horizon"]
        seed = config["env_seed"]
        agent_view_size = config["agent_view_size"]

        self.obstacle_type = Lava
        self.wall_type = Wall
        self.last_done = False
        self.exo_pos = None
        self.exo_obj = None

        super().__init__(
            width=width,
            height=height,
            max_steps=horizon,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=seed,
            agent_view_size=agent_view_size
        )

        self.min_dist_to_goal = 8

        # Actions are discrete integer values
        self.actions = TwoGrids.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        self.reward_decay_ratio = 0.1  # config["reward_decay_ratio"]

    def _gen_grid(self, width, height):
        assert width == 24 and height == 12

        # Create an empty grid
        self.grid = Grid(width, height)

        # # Generate the surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the mid-left
        self.agent_pos = (0, 0)
        self.agent_dir = 0

        self.exo_pos = (12, 0)
        self.exo_dir = 0
        self.exo_obj = Ball()

        # Place a goal square in the mid-right
        self.goal_pos = np.array((6, 4))
        self.put_obj(Goal(), *self.goal_pos)

        self.put_obj(self.exo_obj, *self.exo_pos)

        for pad in [0, 12]:
            self.grid.vert_wall(1 + pad, 1, 10, self.wall_type)
            self.grid.vert_wall(3 + pad, 1, 8, self.wall_type)
            self.grid.vert_wall(10 + pad, 1, 10, self.wall_type)
            self.grid.vert_wall(8 + pad, 3, 6, self.wall_type)
            self.grid.vert_wall(5 + pad, 3, 4, self.wall_type)

            self.grid.horz_wall(4 + pad, 1, 6, self.wall_type)
            self.grid.horz_wall(2 + pad, 10, 8, self.wall_type)
            self.grid.horz_wall(4 + pad, 8, 4, self.wall_type)
            self.grid.horz_wall(6 + pad, 3, 2, self.wall_type)
            self.grid.horz_wall(6 + pad, 6, 1, self.wall_type)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def reset(self):

        self.last_done = False
        return super().reset()

    def valid_agent(self, pos):
        return 0 <= pos[0] < self.width // 2 and 0 <= pos[1] < self.height

    def valid_exo(self, pos):
        return self.width // 2 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def step(self, action):

        if self.last_done:
            # If done then the agent gets stuck
            obs = None
            # obs = self.gen_obs()
            return obs, 0.0, True, {}

        self.step_count += 1

        reward = self._noop_reward()
        done = False

        # Rotate left
        if action == self.actions.left or action == self.actions.left_forward:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right or action == self.actions.right_forward:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        if self.valid_agent(self.front_pos):
            fwd_cell = self.grid.get(*fwd_pos)
            outside = False
        else:
            fwd_cell = None
            outside = True

        # Move forward
        if action == self.actions.left or action == self.actions.right:
            pass
        elif action == self.actions.forward \
                or action == self.actions.left_forward or action == self.actions.right_forward:

            if outside:
                reward = self._lava_reward()
            else:

                if fwd_cell is None or fwd_cell.can_overlap():
                    self.agent_pos = fwd_pos

                if fwd_cell is not None and fwd_cell.type == 'goal':
                    done = True
                    self.agent_pos = fwd_pos
                    reward = self._goal_reward()

                if fwd_cell is not None and fwd_cell.type == 'lava':
                    done = True
                    self.agent_pos = fwd_pos
                    reward = self._lava_reward()
        else:
            assert False, "unknown action %r" % action

        if self.step_count >= self.max_steps:
            done = True

        obs = None
        # obs = self.gen_obs()

        self.move_exo_obj()

        return obs, reward, done, {}

    def move_exo_obj(self):

        # find moveable positions for ball and randomly pick one
        next_exo_positions = [
                (self.exo_pos[0], self.exo_pos[1]),
                (self.exo_pos[0] - 1, self.exo_pos[1]),
                (self.exo_pos[0] + 1, self.exo_pos[1]),
                (self.exo_pos[0], self.exo_pos[1] - 1),
                (self.exo_pos[0], self.exo_pos[1] + 1)
            ]

        valid_positions = [pos for pos in next_exo_positions if self.valid_exo(pos) and self.grid.get(*pos) is None]

        # Free old position of ball
        self.grid.set(*self.exo_pos, None)

        # Pick a random choice
        self.exo_pos = random.choice(valid_positions)

        # Update the object's position
        # self.place_obj(self.exo_obj, self.exo_pos)
        self.put_obj(self.exo_obj, *self.exo_pos)

    def calc_step(self, state, action):

        agent_pos, agent_dir = (state[0], state[1]), state[2]

        # Rotate left
        if action == self.actions.left or action == self.actions.left_forward:
            agent_dir -= 1
            if agent_dir < 0:
                agent_dir += 4

        # Rotate right
        elif action == self.actions.right or action == self.actions.right_forward:
            agent_dir = (agent_dir + 1) % 4

        # Get the position in front of the agent
        if agent_dir == 0:
            fwd_pos = agent_pos[0] + 1, agent_pos[1]
        elif agent_dir == 1:
            fwd_pos = agent_pos[0], agent_pos[1] + 1
        elif agent_dir == 2:
            fwd_pos = agent_pos[0] - 1, agent_pos[1]
        elif agent_dir == 3:
            fwd_pos = agent_pos[0], agent_pos[1] - 1
        else:
            raise AssertionError("Action dir has to be in {0, 1, 2, 3}")

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward
        if action == self.actions.left or action == self.actions.right:
            pass
        elif action == self.actions.forward \
                or action == self.actions.left_forward or action == self.actions.right_forward:

            if fwd_cell is None or fwd_cell.can_overlap():
                agent_pos = fwd_pos

            if fwd_cell is not None and fwd_cell.type == 'goal':
                agent_pos = fwd_pos

            if fwd_cell is not None and fwd_cell.type == 'lava':
                agent_pos = fwd_pos
        else:
            assert False, "unknown action"

        return agent_pos[0], agent_pos[1], agent_dir

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        raise NotImplementedError

    def _noop_reward(self):
        return -0.01

    def _lava_reward(self):
        return -1

    def _goal_reward(self):
        return 1

