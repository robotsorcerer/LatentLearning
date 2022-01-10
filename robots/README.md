### Authors

+ Lekan Molu


### INTRODUCTION

These codes are useful for generating the experiments described in paper <such and such> by <such and such>. These experiments are the robot simulation testbeds  for learning  <such and such> policies.

The steps below are boilerplates to see how a local control law is applied to joints in order to realize a home positioning of the arm.

### BOX2D Prerequisites

+ Box 2D: For running inverted pendulum and a simple double pendulum swing-up task, install the following packages first:

    + `$ sudo apt-get install build-essential python-dev swig python-pygame git`

    + In a separate directory on your system (I prefer Downloads), `$ git clone https://github.com/pybox2d/pybox2d`

    + Then build and install like so:
      ```$ python setup.py build

          # Assuming everything goes well...
          $ sudo python setup.py install
      ```

In addition, we will install other ad-hoc dependencies via pip as follows:

    + ```pip install -r requirements```

+ To run for the [double_pendulum]() experiment, change this line in [main.py](https://github.com/robotsorcerer/LatentLearning/blob/delldevs/robots/main.py#L35) to `double_pendulum`. For some reason, pygame's default options keeps overriding absl-py's FLAGS. So we'll have to resort to manual overrides for now.

Be sure to follow the style of the hyperparams file for the [inverted_pendulum](https://github.com/robotsorcerer/LatentLearning/blob/delldevs/robots/experiments/double_pendulum/hyperparams.py) in order to have reasonable controls.

Run like so for data collection:

```python main.py```

Trajectory Samples will be dumped in the `experiments/double_pendulum/data_files/` folder.



#### RUNNING BOX2D Experiments

    `python.exe .\main.py <experiment_name>`

    where `<experiment_name>` could be any of the following

      + `inverted_pendulum`
      + `double_pendulum`
      + ~~`double_integrator`~~
      + ~~`torobo_robot`~~


##### Structure of Main

    + A General class called `LatentLearner` handles the generic data collection problem for any kind of robot or dynamical system.

    + All dynamical systems are separated by an `agent class`. For example, the pendulum experiments are handled by `agent_box2d` agent -- so named because we use Eric Catto's Box2D environment to visualize the dynamics in the state space as we attempot to control the respective agents to equilibrium.

        + Agents can be found in the [agents](/agents) folder

    **How it works**: Hyperparameters for each experiment are stored in respective folders within the experiment sub-folder. We are required to append the name of an experiment to the calling signature of `main.py`. For example, if we are running the double integrator experiment, we'd call it like so:

    `python main.py double_integrator`.

    Data collection happens in the `run` member function of the class, particularly within the function `self._take_sample(...)`. We pick different initial conditions (since this is a nonlinear system), linearize the dynamics about the equilibrium, and then use a linear quadratic feedback regulator in generating stable trajectories. After we are done with data collection, we can take an iteration of <insert MSR NYC's RL algo> on collected trajectory samples with the function `self._take_iteration(trajectory_samples)` in order to learn <insert type pf policy here>.


### Running ROS Experiments

First install ROS 1.x. I recommend a Ubuntu 18.04 Linux Distro and ROS Melodic LTS.
It's best to pull the `nvidia-docker` and put a `ros` and `gazebo` image on it:

  + Install [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

  + [Dockerfile for ROS Melodic](https://hub.docker.com/layers/amd64/ros/melodic/images/sha256-6c99c80a97d9a6af8b830bd34c028e9a80d42138ab0d4ab9bfa78998c70c0954?context=explore)

  + [Hub page for ROS GAZEBO](https://hub.docker.com/_/gazebo)

Otherwise, if you'd rather have the packages domiciled on your computer, here are
instructions for installing these robot operating system tools on your computer:

  + Install [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)

  + Install [catkin tools](https://catkin-tools.readthedocs.io/en/latest/installing.html)

  + Create your ros workspace: `mkdir -p ros_ws/src && cd ros_ws`

  + From your ros workspace root, carry out the following actions:

    + Install packages dependencies:

      ```
      rosdep install --from-paths src -ry
      ```

    + Build catkin workspace:

      ```
      catkin build
      ```

    + Source the development environment variables:

      ```
      source devel/setup.sh
      ```

#### Launching the Setups

  Launch the gazebo simulation:

    ```
    roslaunch arm_gazebo empty_world.launch
    ```

  + Launch the controller along with RViz:

      ```
      roslaunch arm_control rviz.launch
      ```

  + Launch the MoveIt! move group:
      ```
      roslaunch arm_control moveit.launch
      ```

  You can now use the MoveIt! plugin in Rviz to control the arm.
___

### Alex's MNIST Two

+ Install ROS 2 by following the instructions here:

  - [Installation Instructions](https://docs.ros.org/en/foxy/Tutorials/Creating-Your-First-ROS2-Package.html)

+ Run Mnist Two Examples

   ```colcon build --paths src/mnist_two/```
