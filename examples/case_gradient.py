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
Ftip_ = tf.stack([Ftip]*N, axis=0)

# tv = tf.Variable([1.])
# print("Variables")
# print(tv)
# @tf.function
# def dbg():
    # with tf.GradientTape() as tape:
        # R1_ = tf.eye(3)
        # R21_ = tf.constant([[-1,0,0],[0,0,0],[0,0,0]], tf.float32)
        # R22_ = tf.constant([[0,0,0],[0,0,0],[0,0,-1]], tf.float32)
        # R23_ = tf.constant([[1,0,0],[0,-1,0],[0,0,-1]], tf.float32)
        # R3_ = tf.constant([[0, 0, 1],[1, 0, 0],[0, 1, 0]], tf.float32)
        # inp = tf.stack([R1_, R22_], 0) * tv
        # out = 2*tf_rbdl.SO3_to_so3(inp)
        # norm = tf.math.reduce_mean(out)
    # print("grad")
    # grads = tape.gradient(norm, tv)
    # print(grads)


# tf.config.experimental_run_functions_eagerly(True)
# dbg()
# tf.config.experimental_run_functions_eagerly(False)
# print("Variables")
# print(tv)

def case_fn(x):
    N = tf.shape(x)[0]
    positive_idx = tf.cast(tf.squeeze(tf.where(tf.squeeze(tf.math.greater(x, 0.)))),tf.int32)
    negative_idx = tf.cast(tf.squeeze(tf.where(tf.squeeze(tf.math.less_equal(x, 0.)))),tf.int32)
    def all_positive_case():
        y_positive = x*2.

        return y_positive

    def all_negative_case():
        y_negative = x-2.

        return y_negative

    def some_positive_some_negative_case():
        x_positive = tf.gather(x, positive_idx)
        x_negative = tf.gather(x, negative_idx)

        y_positive = x_positive*2.
        y_negative = x_negative-2.

        y_positive = tf.scatter_nd(tf.expand_dims(positive_idx,1),y_positive,tf.stack([N,1]))
        y_negative = tf.scatter_nd(tf.expand_dims(negative_idx,1),y_negative,tf.stack([N,1]))

        return y_positive + y_negative

    all_positive = tf.math.equal(tf.shape(negative_idx)[0], 0)
    all_negative = tf.math.equal(tf.shape(positive_idx)[0], 0)
    return tf.case([(all_positive, all_positive_case), (all_negative, all_negative_case)], default=some_positive_some_negative_case)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
trainable_variable = tf.Variable([[1.], [-1.], [2.], [-2.]])
print("Before Training")
print(trainable_variable)
@tf.function
def upd():
    with tf.GradientTape() as tape:
        y = case_fn(trainable_variable)
    grad = tape.gradient(y, trainable_variable)
    print("grad")
    print(type(grad))
    print(grad)
    optimizer.apply_gradients(zip([grad], [trainable_variable]))

upd()
print("After Training")
print(trainable_variable)
