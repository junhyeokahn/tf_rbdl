import os
import sys
sys.path.append(os.getcwd())
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import tf_rbdl.tf_rbdl as tf_rbdl
import tf_rbdl.rbdl as rbdl

N = 5 # Batch
M = 500 # Epoch

th = tf.constant([0.1, 0.1, 0.1], tf.float32)
dth = tf.constant([0.1, 0.2, 0.3], tf.float32)
ddth = tf.constant([2, 1.5, 1], tf.float32)
g = tf.constant([0, 0, -9.8], tf.float32)
Ftip = tf.constant([1, 1, 1, 1, 1, 1], tf.float32)
M01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]], np.float32)
M12 = np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0],[0, 0, 0, 1]], np.float32)
M23 = np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197],[0, 0, 1, 0.395], [0, 0, 0, 1]], np.float32)
M34 = np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0.14225], [0, 0, 0, 1]], np.float32)
G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
Glist = [G1.astype(np.float32), G2.astype(np.float32), G3.astype(np.float32)]
Glist_ = [tf.convert_to_tensor(G1, tf.float32), tf.convert_to_tensor(G2, tf.float32), tf.convert_to_tensor(G3, tf.float32)]
Mlist = [M01, M12, M23, M34]
Mlist_ = [tf.convert_to_tensor(M01, tf.float32), tf.convert_to_tensor(M12, tf.float32), tf.convert_to_tensor(M23, tf.float32), tf.convert_to_tensor(M34, tf.float32)]
Slist = [np.array([1, 0, 1,      0, 1,     0], np.float32),
          np.array([0, 1, 0, -0.089, 0,     0], np.float32),
          np.array([0, 1, 0, -0.089, 0, 0.425], np.float32)]
Slist_ = [tf.constant([1, 0, 1,      0, 1,     0], tf.float32),
          tf.constant([0, 1, 0, -0.089, 0,     0], tf.float32),
          tf.constant([0, 1, 0, -0.089, 0, 0.425], tf.float32)]
th_ = tf.stack([th]*N, axis=0)
dth_ = tf.stack([dth]*N, axis=0)
ddth_ = tf.stack([ddth]*N, axis=0)
merged_input_ = tf.concat([th_, dth_, ddth_], axis=1)
Ftip_ = tf.stack([Ftip]*N, axis=0)

tf_rbdl.id_space(tf.expand_dims(th,0), tf.expand_dims(dth,0), tf.expand_dims(ddth,0), g, tf.expand_dims(Ftip,0), Mlist_, Glist_, Slist_)
with tf.device('/CPU:0'):
    tf_rbdl.id_space(tf.expand_dims(th,0), tf.expand_dims(dth,0), tf.expand_dims(ddth,0), g, tf.expand_dims(Ftip,0), Mlist_, Glist_, Slist_)
t1 = time.time()
with tf.device('/CPU:0'):
    for j in tqdm(range(M)):
        for i in range(N):
            rbdl.id_space(th, dth, ddth, g, Ftip, Mlist_, Glist_, Slist_)
t2 = time.time()
print("Single Input (CPU w/ Autograph): ", t2 - t1)

t1 = time.time()
tf.config.experimental_run_functions_eagerly(True)
with tf.device('/CPU:0'):
    for j in tqdm(range(M)):
        for i in range(N):
            rbdl.id_space(th, dth, ddth, g, Ftip, Mlist_, Glist_, Slist_)
tf.config.experimental_run_functions_eagerly(False)
t2 = time.time()
print("Single Input (CPU w/o Autograph): ", t2 - t1)

tf_rbdl.id_space(tf.expand_dims(th,0), tf.expand_dims(dth,0), tf.expand_dims(ddth,0), g, tf.expand_dims(Ftip,0), Mlist_, Glist_, Slist_)
t1 = time.time()
with tf.device('/CPU:0'):
    for i in tqdm(range(M)):
        tf_rbdl.id_space(th_, dth_, ddth_, g, Ftip_, Mlist_, Glist_, Slist_)
t2 = time.time()
print("Batch Input (CPU w/ Autograph): ", t2-t1)

t1 = time.time()
tf.config.experimental_run_functions_eagerly(True)
with tf.device('/CPU:0'):
    for i in tqdm(range(M)):
        tf_rbdl.id_space(th_, dth_, ddth_, g, Ftip_, Mlist_, Glist_, Slist_)
tf.config.experimental_run_functions_eagerly(False)
t2 = time.time()
print("Batch Input (CPU w/o Autograph): ", t2-t1)

t1 = time.time()
for j in tqdm(range(M)):
    for i in range(N):
        rbdl.id_space(th, dth, ddth, g, Ftip, Mlist_, Glist_, Slist_)
t2 = time.time()
print("Single Input (GPU w/ Autograph): ", t2 - t1)

tf.config.experimental_run_functions_eagerly(True)
t1 = time.time()
for j in tqdm(range(M)):
    for i in range(N):
        rbdl.id_space(th, dth, ddth, g, Ftip, Mlist_, Glist_, Slist_)
tf.config.experimental_run_functions_eagerly(False)
t2 = time.time()
print("Single Input (GPU w/o Autograph): ", t2 - t1)

t1 = time.time()
for i in tqdm(range(M)):
    tf_rbdl.id_space(th_, dth_, ddth_, g, Ftip_, Mlist_, Glist_, Slist_)
t2 = time.time()
print("Batch Input (GPU w/ Autograph): ", t2-t1)

t1 = time.time()
tf.config.experimental_run_functions_eagerly(True)
for i in tqdm(range(M)):
    tf_rbdl.id_space(th_, dth_, ddth_, g, Ftip_, Mlist_, Glist_, Slist_)
tf.config.experimental_run_functions_eagerly(False)
t2 = time.time()
print("Batch Input (GPU w/o Autograph): ", t2-t1)
