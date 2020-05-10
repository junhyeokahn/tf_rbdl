import tensorflow as tf

from tf_rbdl.liegroup import *

@tf.function
def fk(M,Blist,Bidlist,theta):
    """
    Forward Kinematics within Body Frame

    Parameters
    ----------
    M (tf.Tensor):
        SE(3) of the end-effector
        (4,4)
    Blist (tf.Tensor):
        Joint screw axes in the end-effector frame when the
        manipulator is at the home position
        (nq_branch,)
    Bidlist (tf.Tensor):
        Joint index related to the end-effector branch
        (nq_branch,)
    theta (tf.Tensor):
        Joint positions
        (N,nq)

    Returns
    ------
    ret (tf.Tensor):
        SE(3) of the End-effector
        (N,4,4)
    """
    N, nq = theta.shape
    nq_branch = Bidlist.shape[0]
    T = tf.tile(tf.expand_dims(M, axis=0), tf.stack([N,1,1],0))
    for i in tf.range(nq_branch):
        tf.autograph.experimental.set_loop_options(shape_invariants=[(T, tf.TensorShape([None,4,4]))])
        T = tf.matmul(T, se3_to_SE3(vec_to_se3(Blist[i] * tf.expand_dims(theta[:,Bidlist[i]], axis=1))))
    return T

@tf.function
def jacobian(Blist,Bidlist,theta):
    """
    Jacobian in the Body Frame

    Parameters
    ----------
    Blist (tf.Tensor):
        Joint screw axes in the end-effector frame when the
        manipulator is at the home position
        (nq_branch,)
    Bidlist (tf.Tensor):
        Joint index related to the end-effector branch
        (nq_branch,)
    theta (tf.Tensor):
        Joint positions
        (N,nq)

    Returns
    ------
    ret (tf.Tensor):
        Body jacobian
        (N,6,nq)
    """
    N, nq = theta.shape
    nq_branch = Bidlist.shape[0]
    Jb = tf.TensorArray(tf.float32, size=nq)
    for i in tf.range(nq_branch):
        Jb = Jb.write(Bidlist[i], tf.tile(tf.expand_dims(Blist[i], axis=0), tf.constant([N,1])))
    T = tf.tile(tf.expand_dims(tf.eye(4), axis=0), tf.constant([N,1,1]))
    for i in tf.range(nq_branch-2,-1,-1):
        tf.autograph.experimental.set_loop_options(shape_invariants=[(T, tf.TensorShape([None,4,4]))])
        T = tf.matmul(T, se3_to_SE3(vec_to_se3(Blist[i+1] * -tf.expand_dims(theta[:,Bidlist[i+1]],axis=1))))
        Jb = Jb.write(Bidlist[i], tf.squeeze(tf.matmul(adjoint(T), tf.expand_dims(Blist[i], axis=1)), axis=2))
    return tf.transpose(Jb.stack(), perm=[1,2,0])
