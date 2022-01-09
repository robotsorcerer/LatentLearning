__all__ = ["InvertedPendulum"]

import h5py
import pygame
import Box2D as b2
import numpy as np
from agents.box2d.framework import Framework
from agents.box2d.settings import fwSettings
from utility import rad2deg

class InvertedPendulum(Framework):
    name = "Inverted Pendulum"
    def __init__(self, x0, target, render, integrator):
        self.render = render
        # if self.render:
        super(InvertedPendulum, self).__init__(render)
        # else:
        #     self.world = b2.b2World(gravity=(0, -10), doSleep=True)

        #set body gravity to zero
        self.world.gravity = (0.0, 0.0)

        self.fixture_length = 10
        self.x0 = x0
        self.gravity = 10

        rectangle_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(box=(.5, self.fixture_length)),
            density=1.0,
            friction=1,
        )
        square_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(box=(2, 2)),
            density=1.0,
            friction=1,
        )
        self.base = self.world.CreateBody(
            position=(0, 20),
            fixtures=square_fixture,
        )

        self.body1 = self.world.CreateDynamicBody(
            position=(0, 4),
            fixtures=rectangle_fixture,
            angle=b2.b2_pi,
        )

        self.target1 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 0),
            angle=b2.b2_pi,
        )

        self.joint1 = self.world.CreateRevoluteJoint(
            bodyA=self.base,
            bodyB=self.body1,
            localAnchorA=(0, 0),
            localAnchorB=(0, self.fixture_length),
            enableMotor=True,
            maxMotorTorque=400,
            enableLimit=False,
        )

        self.set_joint_angles(self.body1, x0[0])
        self.set_joint_angles(self.target1, target[0])
        self.target1.active = False

        self.joint1.motorSpeed = x0[1]

        """
        Control parameters for LQR Linearization.
        """
        # Given the dynamics from the ODE, linearize about an equilibrium position
        self.A = np.array(
                            [[ 0 ,1],
                             [self.body1.mass*self.gravity*self.fixture_length/self.body1.inertia, 0]
                            ]
                            )
        self.B = np.array( [[0], [1/self.body1.inertia]] )
        self.C = np.array([[1, 0]])
        self.D = [0]
        self.xe = np.array([0., 0.]) # equilibrium position in absolute coordinates

        # Quadratic regulator matrix paramters
        self.Qx1 = np.diag([1, 1])
        self.Qu1a = np.diag([1])

        self.integrator = lambda x: integrator(x, self.body1.mass, self.fixture_length)

    def set_joint_angles(self, body1, angle1):
        """ Converts the given absolute angle of the arms to joint angles"""
        pos = self.base.GetWorldPoint((0, 0))
        body1.angle = angle1 + np.pi
        new_pos = body1.GetWorldPoint((0, self.fixture_length))
        body1.position += pos - new_pos
        pos = body1.GetWorldPoint((0, -4.5))


    def run(self, action=None):
        """Initiates the first time step
        """
        if self.render:
            super(InvertedPendulum, self).run()
        else:
            self.run_next(action)

    def run_next(self, action):
        """Moves forward in time one step. Calls the renderer if applicable."""
        if self.render:
            super(InvertedPendulum, self).run_next(action)
        else:
            if action is not None:
                self.joint1.motorSpeed = action[0]
            self.world.Step(1.0 / fwSettings.hz, fwSettings.velocityIterations,
                            fwSettings.positionIterations)

    def Step(self, settings, action):
        """Moves forward in time one step. Called by the renderer"""
        self.joint1.motorSpeed = action[0]

        super(InvertedPendulum, self).Step(settings)
        # self.Print(f"Joint state = {rad2deg(self.joint1.angle):.4f} deg, speed = {self.joint1.motorSpeed:.4f} m/s." )

    def reset_world(self):
        """Returns the world to its intial state"""
        self.world.ClearForces()
        self.joint1.motorSpeed = 0
        self.body1.linearVelocity = (0, 0)
        self.body1.angularVelocity = 0
        self.set_joint_angles(self.body1, self.x0[0])


    def get_state(self):
        """Retrieves the state of the point mass"""
        state = {'JOINT_ANGLES': np.array([self.joint1.angle]),
                 'JOINT_VELOCITIES': np.array([self.joint1.speed]),
                 'END_EFFECTOR_POINTS': np.append(np.array(self.body1.position),[0]), 
                # https://github.com/5h00T/avoid_game_env/blob/2b4f35791cffde417d1020ee7384268da9340db0/gym_avoid_game/envs/avoid_game_env.py#L34
                'OBSERVATIONS': np.asarray(pygame.surfarray.array3d(self.screen).T), # will be 3 X 480 X 640
                }
        # print('state: ', state)
        return state
    
    # def save_screen(self, fname):
    #     """Save screenshot for onward processing by latent s[ace learner."""
    #     if not self.render:
    #         return ValueError("You are running pygame without video device.")
    #     # pygame.image.save(self.screen, fname)
    #     img_arr = self.get_state()

    def save_iter(self, sample_grp, t):
        """Save screenshot for onward processing by latent s[ace learner."""
        if not self.render:
            return ValueError("You are running pygame without video device.")
        # pygame.image.save(self.screen, fname)
        # with h5py.File(fname, 'a') as h5file:
        #     cond_grp = h5file.create_group(f'condition_{cond:0>2}')


        state =  self.get_state()
        for k,v in state.items():
            if k=='OBSERVATIONS':
                sample_grp.create_dataset(f"{k}_{t:0>3}", data=v, compression="gzip")#, dtype=v.dtype)
            else:
                sample_grp.create_dataset(f"{k}_{t:0>3}", data=v)#, dtype=v.dtype)

