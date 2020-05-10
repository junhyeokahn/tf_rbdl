import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())

from tf_rbdl.kinematics import *

# Blist = tf.TensorArray(tf.float32, size=3, clear_after_read=False)
# for i in range(3):
    # Blist = Blist.write(i, tf.zeros(6))
# Idlist = tf.TensorArray(tf.int32, size=3, clear_after_read=False)
# for i in range(3):
    # Idlist = Idlist.write(i, i)
# M = tf.eye(4)
# thetalist = tf.zeros((2,5))

# T = fk(M, Blist, Idlist, thetalist)
# print(T)
# J = jacobian(Blist, Idlist, thetalist)
# print(J)
# tf.config.experimental_run_functions_eagerly(False)


@tf.function 
def f(x): 
    for i in range(3):
        x.read(i)
    x.read(0)
    return 0
 
x = tf.TensorArray(tf.int32, size=3,clear_after_read=False)
for i in range(3):
    x = x.write(i,i)
f(x) 


