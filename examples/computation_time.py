import os
import sys
sys.path.append(os.getcwd())
import time

from mujoco_py import load_model_from_path, MjSim, functions, MjViewer
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tf_rbdl.dynamics import *
from tf_rbdl.utils import *

xml_path = os.getcwd()+'/examples/assets/my_hopper.xml'
ee_list = ['foot_sole']
N = 256*100 # Batch
M = 1 # Epoch

# ==============================================================================
# Mujoco
# ==============================================================================
mujoco_model = load_model_from_path(xml_path)
m = MjSim(mujoco_model)
m.forward()

q = np.random.uniform(-1., 1., (N,m.model.nq))
qdot = np.random.uniform(-1., 1., (N,m.model.nq))

sim_state=m.get_state()
t1 = time.time()
for i in tqdm(range(M)):
    for j in range(N):
        for k in range(m.model.nq):
            sim_state.qpos[k] = q[j,k]
            sim_state.qvel[k] = qdot[j,k]
        m.set_state(sim_state)
        m.forward()
        M_ = np.zeros(m.model.nv * m.model.nv)
        functions.mj_fullM(m.model, M_, m.data.qM)
t2 = time.time()
print("Mujoco: ", t2 - t1)

# ==============================================================================
# tf_rbdl
# ==============================================================================
ic = initial_config_from_mjcf(xml_path, ee_list, verbose=False)
mass_matrix(tf.convert_to_tensor(q,tf.float32),ic['pidlist'], ic['Mlist'], ic['Glist'], ic['Slist'])
t1 = time.time()
for i in tqdm(range(M)):
    mass_matrix(tf.convert_to_tensor(q,tf.float32),ic['pidlist'], ic['Mlist'], ic['Glist'], ic['Slist'])
t2 = time.time()
print("tf_rbdl with GPU: ", t2 - t1)

with tf.device('/CPU:0'):
    mass_matrix(tf.convert_to_tensor(q,tf.float32),ic['pidlist'], ic['Mlist'], ic['Glist'], ic['Slist'])
t1 = time.time()
with tf.device('/CPU:0'):
    for i in tqdm(range(M)):
        mass_matrix(tf.convert_to_tensor(q,tf.float32),ic['pidlist'], ic['Mlist'], ic['Glist'], ic['Slist'])
t2 = time.time()
print("tf_rbdl with CPU: ", t2 - t1)
