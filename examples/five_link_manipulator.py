from tqdm import tqdm
import numpy as np
import pybullet as p
from mujoco_py import load_model_from_path, MjSim, functions, MjViewer
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

pybullet_model = p.loadURDF(os.getcwd()+'/examples/assets/five_link_manipulator.urdf', useFixedBase=0)
p.stepSimulation()

for i in range(5):
    print("Link {}".format(i))
    print("COM WORLD POS")
    print(p.getLinkState(0, i)[0])
    print("COM WORLD ORI")
    print(p.getLinkState(0, i)[1])
    print("LINK POS")
    print(p.getLinkState(0, i)[4])
    print("LINK ORI")
    print(p.getLinkState(0, i)[5])
    print("MASS")
    print(p.getDynamicsInfo(0, i)[0])
    print("INERTIA")
    print(p.changeDynamics(0, i, localInertiaDiagonal=[1,2,3]))
    print(p.getDynamicsInfo(0, i)[2])


mujoco_model = load_model_from_path(os.getcwd()+'/examples/assets/five_link_manipulator.xml')
m = MjSim(mujoco_model)
m.forward()

pMass = p.calculateMassMatrix(0, [0,0,0,0,0])
mMass = np.zeros(m.model.nv * m.model.nv)
functions.mj_fullM(m.model, mMass, m.data.qM)
print("Bullet Mass Matrix:")
print(np.array(pMass))
print("Mujoco Mass Matrix")
print(mMass.reshape(5,5))
__import__('ipdb').set_trace()
exit()
