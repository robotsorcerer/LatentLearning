import mujoco_py
from os.path import expanduser, join

mj_path = join(expanduser("~"), "~/mujoco/mujoco-211")
# mj_path = mujoco_py.utils.discover_mujoco()
xml_path = join(mj_path, 'model', 'humanoid', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

sim.step()
print(sim.data.qpos)