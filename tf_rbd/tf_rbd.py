import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import copy

@tf.function
def near_zero(z):
    """
    Checks if the scalar is small enough to be neglected.

    Parameters
    ----------
    z (tf.Tensor):
        N x 1

    Returns
    -------
    ret (tf.Tensor):
        N x 1
    """
    return tf.less(tf.math.abs(z), 1e-3)

@tf.function
def normalize_vector(V):
    """
    Normalize vector

    Parameters
    ----------
    V (tf.Tensor):
        N x M

    Returns
    -------
    ret (tf.Tensor):
        N x M
    """
    return tf.linalg.normalize(V, axis=1)[0]

@tf.function
def SO3_inv(R):
    """
    Inverse of SO(3)

    Parameters
    ----------
    R (tf.Tensor):
        SO(3)
        N x 3 x 3

    Returns
    -------
    ret (tf.Tensor):
        Inverse of the input SO(3)
        N x 3 x 3
    """
    return tf.transpose(R, perm=[0,2,1])

@tf.function
def vec_to_so3(omg):
    """
    Convert angular velocity to so(3)

    Parameters
    ----------
    omg (tf.Tensor):
        Angular velocity
        N x 3

    Returns
    -------
    ret (tf.Tensor):
        so(3)
        N x 3 x 3
    """
    N = tf.shape(omg)[0]
    omgx = tf.expand_dims(omg[:,0], axis=1)
    omgy = tf.expand_dims(omg[:,1], axis=1)
    omgz = tf.expand_dims(omg[:,2], axis=1)
    zeros = tf.zeros((N, 1))
    return tf.concat([tf.stack([zeros, -omgz, omgy], axis=2), tf.stack([omgz, zeros, -omgx], axis=2), tf.stack([-omgy, omgx, zeros], axis=2)], axis=1)

@tf.function
def so3_to_vec(so3mat):
    """
    Convert so(3) to angular velocity

    Parameters
    ----------
    so3mat (tf.Tensor):
        so(3)
        N x 3 x 3

    Returns
    -------
    ret (tf.Tensor):
        Angular velocity
        N x 3
    """
    return tf.stack([so3mat[:,2,1], so3mat[:,0,2], so3mat[:,1,0]], axis=1)

@tf.function
def angvel_to_axis_ang(expc3):
    """
    Convert a 3d-vector of exponential coordinates for rotation to an unit
    rotation axis omghat and the corresponding rotation angle theta.

    Parameters
    ----------
    expc3 (tf.Tensor):
        exponential coordinates for rotation
        N x 3

    Returns
    -------
    ret (tuple):
        (axis, angle)
        (N x 3, N x 1)
    """
    return tf.linalg.normalize(expc3, axis=1)

@tf.function
def so3_to_SO3(so3mat):
    """
    Convert so(3) to SO(3)

    Parameters
    ----------
    so3mat (tf.Tensor):
        so(3)
        N x 3 x 3

    Returns
    ------
    ret (tf.Tensor):
        SO(3)
        N x 3 x 3
    """

    N = tf.shape(so3mat)[0]
    omgtheta = so3_to_vec(so3mat)
    norm_omgtheta = tf.norm(omgtheta, axis=1)
    zero_idx = tf.cast(tf.squeeze(tf.where(near_zero(norm_omgtheta)), axis=1), tf.int32)
    non_zero_idx = tf.cast(tf.squeeze(tf.where(tf.math.logical_not(near_zero(norm_omgtheta))), axis=1), tf.int32)

    def f1():
        theta = tf.expand_dims(angvel_to_axis_ang(omgtheta)[1], axis=1)
        omgmat = so3mat / theta
        return tf.eye(3) + tf.sin(theta) * omgmat + (1 - tf.cos(theta)) * tf.matmul(omgmat,omgmat)
    def f2():
        return tf.tile( tf.expand_dims(tf.eye(3), axis=0), tf.stack([N, 1, 1]) )
    def f3():
        zero_case_input = tf.gather(so3mat, zero_idx)
        zero_case_output = tf.tile( tf.expand_dims(tf.eye(3), axis=0), tf.stack([tf.shape(zero_idx)[0], 1, 1], 0))

        non_zero_case_input = tf.gather(so3mat, non_zero_idx)
        omgtheta2 = so3_to_vec(non_zero_case_input)
        theta = tf.expand_dims(angvel_to_axis_ang(omgtheta2)[1], axis=1)
        omgmat = non_zero_case_input / theta
        non_zero_case_output = tf.eye(3) + tf.sin(theta) * omgmat + (1 - tf.cos(theta)) * tf.matmul(omgmat,omgmat)

        u1 = tf.scatter_nd(tf.expand_dims(zero_idx, axis=1), zero_case_output, tf.stack([N, 3, 3]))
        u2 = tf.scatter_nd(tf.expand_dims(non_zero_idx, axis=1), non_zero_case_output, tf.stack([N, 3, 3]))
        return u1 + u2

    return tf.case([(tf.math.equal(tf.shape(zero_idx)[0], 0), f1), (tf.math.equal(tf.shape(non_zero_idx)[0], 0), f2)], default=f3)

@tf.function
def R_to_omg_(R):
    N = tf.shape(R)[0]
    b_1 = tf.math.logical_not(near_zero(1 + R[:,2,2]))
    b_2 = tf.math.logical_not(near_zero(1 + R[:,1,1]))
    idx_1 = tf.cast(tf.squeeze(tf.where(b_1), axis=1), tf.int32)
    idx_2 = tf.cast(tf.squeeze(tf.where(tf.math.logical_and(tf.math.logical_not(b_1), b_2)), axis=1), tf.int32)
    idx_3 = tf.cast(tf.squeeze(tf.where(tf.math.logical_not(tf.math.logical_or(b_1, b_2))), axis=1), tf.int32)

    def g1():
        omg = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R[:,2,2]))), axis=1) * tf.stack([R[:,0,2], R[:,1,2], 1+R[:,2,2]], axis=1)
        return omg
    def g2():
        omg = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R[:,1,1]))), axis=1) * tf.stack([R[:,0,1], 1+R[:,1,1], R[:,2,1]], axis=1)
        return omg
    def g3():
        omg = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R[:,0,0]))), axis=1) * tf.stack([1+R[:,0,0], R[:,1,0], R[:,2,0]], axis=1)
        return omg
    def g12():
        R_1 = tf.gather(R, idx_1)
        omg_1 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_1[:,2,2]))), axis=1) * tf.stack([R_1[:,0,2], R_1[:,1,2], 1+R_1[:,2,2]], axis=1)
        omg_1 = tf.scatter_nd(tf.expand_dims(idx_1, axis=1), omg_1, tf.stack([N, 3]))

        R_2 = tf.gather(R, idx_2)
        omg_2 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_2[:,1,1]))), axis=1) * tf.stack([R_2[:,0,1], 1+R_2[:,1,1], R_2[:,2,1]], axis=1)
        omg_2 = tf.scatter_nd(tf.expand_dims(idx_2, axis=1), omg_2, tf.stack([N, 3]))

        return omg_1 + omg_2
    def g13():
        R_1 = tf.gather(R, idx_1)
        omg_1 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_1[:,2,2]))), axis=1) * tf.stack([R_1[:,0,2], R_1[:,1,2], 1+R_1[:,2,2]], axis=1)
        omg_1 = tf.scatter_nd(tf.expand_dims(idx_1, axis=1), omg_1, tf.stack([N, 3]))

        R_3 = tf.gather(R, idx_3)
        omg_3 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_3[:,0,0]))), axis=1) * tf.stack([1+R_3[:,0,0], R_3[:,1,0], R_3[:,2,0]], axis=1)
        omg_3 = tf.scatter_nd(tf.expand_dims(idx_3, axis=1), omg_3, tf.stack([N, 3]))

        return omg_1 + omg_3
    def g23():
        R_2 = tf.gather(R, idx_2)
        omg_2 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_2[:,1,1]))), axis=1) * tf.stack([R_2[:,0,1], 1+R_2[:,1,1], R_2[:,2,1]], axis=1)
        omg_2 = tf.scatter_nd(tf.expand_dims(idx_2, axis=1), omg_2, tf.stack([N, 3]))

        R_3 = tf.gather(R, idx_3)
        omg_3 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_3[:,0,0]))), axis=1) * tf.stack([1+R_3[:,0,0], R_3[:,1,0], R_3[:,2,0]], axis=1)
        omg_3 = tf.scatter_nd(tf.expand_dims(idx_3, axis=1), omg_3, tf.stack([N, 3]))

        return omg_2 + omg_3
    def g123():
        R_1 = tf.gather(R, idx_1)
        omg_1 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_1[:,2,2]))), axis=1) * tf.stack([R_1[:,0,2], R_1[:,1,2], 1+R_1[:,2,2]], axis=1)
        omg_1 = tf.scatter_nd(tf.expand_dims(idx_1, axis=1), omg_1, tf.stack([N, 3]))

        R_2 = tf.gather(R, idx_2)
        omg_2 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_2[:,1,1]))), axis=1) * tf.stack([R_2[:,0,1], 1+R_2[:,1,1], R_2[:,2,1]], axis=1)
        omg_2 = tf.scatter_nd(tf.expand_dims(idx_2, axis=1), omg_2, tf.stack([N, 3]))

        R_3 = tf.gather(R, idx_3)
        omg_3 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_3[:,0,0]))), axis=1) * tf.stack([1+R_3[:,0,0], R_3[:,1,0], R_3[:,2,0]], axis=1)
        omg_3 = tf.scatter_nd(tf.expand_dims(idx_3, axis=1), omg_3, tf.stack([N, 3]))

        return omg_1 + omg_2 + omg_3

    no_idx_1 = tf.math.equal(tf.shape(idx_1)[0], 0)
    yes_idx_1 = tf.math.logical_not(no_idx_1)
    no_idx_2 = tf.math.equal(tf.shape(idx_2)[0], 0)
    yes_idx_2 = tf.math.logical_not(no_idx_2)
    no_idx_3 = tf.math.equal(tf.shape(idx_3)[0], 0)
    yes_idx_3 = tf.math.logical_not(no_idx_3)

    pred_fn_1 = tf.math.logical_and(tf.math.logical_and((yes_idx_1), (no_idx_2)), (no_idx_3))
    pred_fn_2 = tf.math.logical_and(tf.math.logical_and((no_idx_1), (yes_idx_2)), (no_idx_3))
    pred_fn_3 = tf.math.logical_and(tf.math.logical_and((no_idx_1), (no_idx_2)), (yes_idx_3))
    pred_fn_12 = tf.math.logical_and(tf.math.logical_and((yes_idx_1), (yes_idx_2)), (no_idx_3))
    pred_fn_13 = tf.math.logical_and(tf.math.logical_and((yes_idx_1), (no_idx_2)), (yes_idx_3))
    pred_fn_23 = tf.math.logical_and(tf.math.logical_and((no_idx_1), (yes_idx_2)), (yes_idx_3))
    pred_fn_123 = tf.math.logical_and(tf.math.logical_and((yes_idx_1), (yes_idx_2)), (yes_idx_3))

    return tf.case([(pred_fn_1, g1), (pred_fn_2, g2), (pred_fn_3, g3), (pred_fn_12, g12), (pred_fn_13, g13), (pred_fn_23, g23), (pred_fn_123, g123)])

@tf.function
def SO3_to_so3(R):
    """
    Convert SO(3) to so(3)

    Parameters
    ----------
    R (tf.Tensor):
        SO(3)
        N x 3 x 3

    Returns
    ------
    ret (tf.Tensor):
        so(3)
        N x 3 x 3
    """

    N = tf.shape(R)[0]
    b_1 = near_zero(tf.norm(R - tf.eye(3), axis=[-2,-1]))
    b_2 = near_zero(tf.linalg.trace(R) + 1)
    idx_1 = tf.cast(tf.squeeze(tf.where(b_1), axis=1), tf.int32)
    idx_2 = tf.cast(tf.squeeze(tf.where(b_2), axis=1), tf.int32)
    idx_3 = tf.cast(tf.squeeze(tf.where(tf.math.logical_not(tf.math.logical_or(b_1, b_2))), axis=1), tf.int32)

    def f1():
        return tf.zeros((N, 3, 3))
    def f2():
        omg = R_to_omg_(R)
        return vec_to_so3(np.pi*omg)
    def f3():
        acosinput = (tf.linalg.trace(R) - 1) / 2.0
        acosinput = tf.clip_by_value(acosinput, clip_value_min=-1., clip_value_max=1.)
        theta = tf.math.acos(acosinput)
        return tf.expand_dims(tf.expand_dims(theta / 2.0 / tf.sin(theta), axis=1), axis=1) * (R - tf.transpose(R, perm=[0,2,1]))
    def f12():
        R_1 = tf.gather(R, idx_1)
        out_1 = tf.zeros((tf.shape(idx_1)[0], 3, 3))
        out_1 = tf.scatter_nd(tf.expand_dims(idx_1, axis=1), out_1, tf.stack([N, 3, 3]))

        R_2 = tf.gather(R, idx_2)
        omg_2 = R_to_omg_(R_2)
        out_2 = vec_to_so3(np.pi*omg_2)
        out_2 = tf.scatter_nd(tf.expand_dims(idx_2, axis=1), out_2, tf.stack([N, 3, 3]))

        return out_1 + out_2
    def f13():
        R_1 = tf.gather(R, idx_1)
        out_1 = tf.zeros((tf.shape(idx_1)[0], 3, 3))
        out_1 = tf.scatter_nd(tf.expand_dims(idx_1, axis=1), out_1, tf.stack([N, 3, 3]))

        R_3 = tf.gather(R, idx_3)
        acosinput = (tf.linalg.trace(R_3) - 1) / 2.0
        acosinput = tf.clip_by_value(acosinput, clip_value_min=-1., clip_value_max=1.)
        theta = tf.math.acos(acosinput)
        out_3 = tf.expand_dims(tf.expand_dims(theta / 2.0 / tf.sin(theta), axis=1), axis=1) * (R_3 - tf.transpose(R_3, perm=[0,2,1]))
        out_3 = tf.scatter_nd(tf.expand_dims(idx_3, axis=1), out_3, tf.stack([N, 3, 3]))

        return out_1 + out_3
    def f23():
        R_2 = tf.gather(R, idx_2)
        omg_2 = R_to_omg_(R_2)
        out_2 = vec_to_so3(np.pi*omg_2)
        out_2 = tf.scatter_nd(tf.expand_dims(idx_2, axis=1), out_2, tf.stack([N, 3, 3]))

        R_3 = tf.gather(R, idx_3)
        acosinput = (tf.linalg.trace(R_3) - 1) / 2.0
        acosinput = tf.clip_by_value(acosinput, clip_value_min=-1., clip_value_max=1.)
        theta = tf.math.acos(acosinput)
        out_3 = tf.expand_dims(tf.expand_dims(theta / 2.0 / tf.sin(theta), axis=1), axis=1) * (R_3 - tf.transpose(R_3, perm=[0,2,1]))
        out_3 = tf.scatter_nd(tf.expand_dims(idx_3, axis=1), out_3, tf.stack([N, 3, 3]))

        return out_2 + out_3
    def f123():
        R_1 = tf.gather(R, idx_1)
        out_1 = tf.zeros((tf.shape(idx_1)[0], 3, 3))
        out_1 = tf.scatter_nd(tf.expand_dims(idx_1, axis=1), out_1, tf.stack([N, 3, 3]))

        R_2 = tf.gather(R, idx_2)
        omg_2 = R_to_omg_(R_2)
        out_2 = vec_to_so3(np.pi*omg_2)
        out_2 = tf.scatter_nd(tf.expand_dims(idx_2, axis=1), out_2, tf.stack([N, 3, 3]))

        R_3 = tf.gather(R, idx_3)
        acosinput = (tf.linalg.trace(R_3) - 1) / 2.0
        acosinput = tf.clip_by_value(acosinput, clip_value_min=-1., clip_value_max=1.)
        theta = tf.math.acos(acosinput)
        out_3 = tf.expand_dims(tf.expand_dims(theta / 2.0 / tf.sin(theta), axis=1), axis=1) * (R_3 - tf.transpose(R_3, perm=[0,2,1]))
        out_3 = tf.scatter_nd(tf.expand_dims(idx_3, axis=1), out_3, tf.stack([N, 3, 3]))

        return out_1 + out_2 + out_3

    no_idx_1 = tf.math.equal(tf.shape(idx_1)[0], 0)
    yes_idx_1 = tf.math.logical_not(no_idx_1)
    no_idx_2 = tf.math.equal(tf.shape(idx_2)[0], 0)
    yes_idx_2 = tf.math.logical_not(no_idx_2)
    no_idx_3 = tf.math.equal(tf.shape(idx_3)[0], 0)
    yes_idx_3 = tf.math.logical_not(no_idx_3)

    pred_fn_1 = tf.math.logical_and(tf.math.logical_and((yes_idx_1), (no_idx_2)), (no_idx_3))
    pred_fn_2 = tf.math.logical_and(tf.math.logical_and((no_idx_1), (yes_idx_2)), (no_idx_3))
    pred_fn_3 = tf.math.logical_and(tf.math.logical_and((no_idx_1), (no_idx_2)), (yes_idx_3))
    pred_fn_12 = tf.math.logical_and(tf.math.logical_and((yes_idx_1), (yes_idx_2)), (no_idx_3))
    pred_fn_13 = tf.math.logical_and(tf.math.logical_and((yes_idx_1), (no_idx_2)), (yes_idx_3))
    pred_fn_23 = tf.math.logical_and(tf.math.logical_and((no_idx_1), (yes_idx_2)), (yes_idx_3))
    pred_fn_123 = tf.math.logical_and(tf.math.logical_and((yes_idx_1), (yes_idx_2)), (yes_idx_3))

    return tf.case([(pred_fn_1, f1), (pred_fn_2, f2), (pred_fn_3, f3), (pred_fn_12, f12), (pred_fn_13, f13), (pred_fn_23, f23), (pred_fn_123, f123)])

@tf.function
def Rp_to_SE3(R,p):
    """
    Convert R, p pair to SE3

    Parameters
    ----------
    R (tf.Tensor):
        SO(3)
        N x 3 x 3
    p (tf.Tensor):
        N x 3

    Returns
    ------
    ret (tf.Tensor):
        SE(3)
        N x 4 x 4
    """
    N = tf.shape(R)[0]
    Rp = tf.concat([R, tf.expand_dims(p, axis=2)], axis=2)
    last_row = tf.tile(tf.constant([[[0.,0.,0.,1.]]], tf.float32), tf.stack([N, 1, 1]))
    return tf.concat([Rp, last_row], axis=1)

@tf.function
def SE3_to_Rp(T):
    """
    Convert SE3 to R, p

    Parameters
    ----------
    T (tf.Tensor):
        SE(3)
        N x 4 x 4

    Returns
    ------
    R (tf.Tensor):
        SO(3)
        N x 3 x 3
    p (tf.Tensor):
        translation
        N x 3
    """
    return T[:,0:3,0:3], T[:,0:3,3]

@tf.function
def SE3_inv(T):
    """
    Inverse of SE3

    Parameters
    ----------
    T (tf.Tensor):
        SE(3)
        N x 4 x 4

    Returns
    ------
    ret (tf.Tensor):
        Inverse of the SE(3) input
        N x 4 x 4
    """
    R,p = SE3_to_Rp(T)
    Rt = tf.transpose(R, perm=[0,2,1])
    rotated_p = tf.squeeze(-tf.matmul(Rt, tf.expand_dims(p, axis=2)), axis=2)
    return Rp_to_SE3(Rt, rotated_p)

@tf.function
def vec_to_se3(V):
    """
    Take a 6d spatial velocity and convert it to se(3)

    Parameters
    ----------
    V (tf.Tensor):
        spatial velocity (or twist)
        N x 6

    Returns
    ------
    ret (tf.Tensor):
        se(3)
        N x 4 x 4
    """
    N = tf.shape(V)[0]
    so3 = vec_to_so3(V[:,0:3])
    last_row = tf.tile(tf.constant([[[0.,0.,0.,0.]]], tf.float32), tf.stack([N, 1, 1]))
    return tf.concat([tf.concat([so3, tf.expand_dims(V[:,3:6], axis=2)], axis=2), last_row], axis=1)

@tf.function
def se3_to_vec(se3mat):
    """
    Take a se(3) and convert it to 6d spatial velocity

    Parameters
    ----------
    ret (tf.Tensor):
        se(3)
        N x 4 x 4

    Returns
    ------
    V (tf.Tensor):
        spatial velocity (or twist)
        N x 6
    """
    return tf.stack([se3mat[:,2,1],se3mat[:,0,2],se3mat[:,1,0],se3mat[:,0,3],se3mat[:,1,3],se3mat[:,2,3]], axis=1)

@tf.function
def adjoint(T):
    """
    Take a SE(3) and compute corresponding adjoint operator

    Parameters
    ----------
    T (tf.Tensor):
        SE(3)
        N x 4 x 4

    Returns
    ------
    ret (tf.Tensor):
        Adjoint operator
        N x 6 x 6
    """
    N = tf.shape(T)[0]
    R, p = SE3_to_Rp(T)
    first_three_row = tf.concat([R, tf.zeros((N,3,3))], axis=2)
    p_cross_R = tf.matmul(vec_to_so3(p), R)
    last_three_row = tf.concat([p_cross_R, R], axis=2)
    return tf.concat([first_three_row, last_three_row], axis=1)

@tf.function
def se3_to_SE3(se3mat):
    """
    Take a se(3) and convert it to SE(3)

    Parameters
    ----------
    se3mat (tf.Tensor):
        se(3)
        N x 4 x 4

    Returns
    ------
    ret (tf.Tensor):
        SE(3)
        N x 3 x 3
    """
    N = tf.shape(se3mat)[0]
    omgtheta = so3_to_vec(se3mat[:,0:3,0:3])
    b_1 = near_zero(tf.norm(omgtheta, axis=1))
    b_2 = tf.math.logical_not(b_1)
    idx_1 = tf.cast(tf.squeeze(tf.where(b_1), axis=1), tf.int32)
    idx_2 = tf.cast(tf.squeeze(tf.where(b_2), axis=1), tf.int32)

    def f1():
        return Rp_to_SE3( tf.tile(tf.expand_dims(tf.eye(3), axis=0), tf.stack([N, 1, 1])), se3mat[:,0:3,3] )
    def f2():
        theta = angvel_to_axis_ang(omgtheta)[1]
        omgmat = se3mat[:,0:3,0:3] / tf.expand_dims(theta, axis=1)
        R = so3_to_SO3(se3mat[:,0:3,0:3])
        a = tf.tile(tf.expand_dims(tf.eye(3), axis=0), tf.stack([N, 1, 1])) * tf.expand_dims(theta, axis=1)
        b = tf.expand_dims((1-tf.cos(theta)), axis=1) * omgmat
        c = tf.expand_dims(theta - tf.sin(theta), axis=1) * tf.matmul(omgmat,omgmat)
        d = a+b+c
        p = tf.squeeze(tf.matmul(d, tf.expand_dims(se3mat[:,0:3,3], axis=2)) / tf.expand_dims(theta, axis=1), axis=2)
        return Rp_to_SE3(R, p)
    def f12():
        se3_1 = tf.gather(se3mat, idx_1)
        out_1 = Rp_to_SE3( tf.tile(tf.expand_dims(tf.eye(3), axis=0), tf.stack([tf.shape(se3_1)[0], 1, 1])), se3_1[:,0:3,3] )
        out_1 = tf.scatter_nd(tf.expand_dims(idx_1, axis=1), out_1, tf.stack([N, 4, 4]))

        se3_2 = tf.gather(se3mat, idx_2)
        omgtheta_2 = tf.gather(omgtheta, idx_2)
        theta = angvel_to_axis_ang(omgtheta_2)[1]
        omgmat = se3_2[:,0:3,0:3] / tf.expand_dims(theta, axis=1)
        R = so3_to_SO3(se3_2[:,0:3,0:3])
        a = tf.tile(tf.expand_dims(tf.eye(3), axis=0), tf.stack([tf.shape(se3_2)[0], 1, 1])) * tf.expand_dims(theta, axis=1)
        b = tf.expand_dims((1-tf.cos(theta)), axis=1) * omgmat
        c = tf.expand_dims(theta - tf.sin(theta), axis=1) * tf.matmul(omgmat,omgmat)
        d = a+b+c
        p = tf.squeeze(tf.matmul(d, tf.expand_dims(se3_2[:,0:3,3], axis=2)) / tf.expand_dims(theta, axis=1), axis=2)
        out_2 = Rp_to_SE3(R, p)
        out_2 = tf.scatter_nd(tf.expand_dims(idx_2, axis=1), out_2, tf.stack([N, 4, 4]))

        return out_1 + out_2

    no_idx_1 = tf.math.equal(tf.shape(idx_1)[0], 0)
    yes_idx_1 = tf.math.logical_not(no_idx_1)
    no_idx_2 = tf.math.equal(tf.shape(idx_2)[0], 0)
    yes_idx_2 = tf.math.logical_not(no_idx_2)

    pred_fn_1 = tf.math.logical_and((yes_idx_1), (no_idx_2))
    pred_fn_2 = tf.math.logical_and((no_idx_1), (yes_idx_2))
    pred_fn_12 = tf.math.logical_and((yes_idx_1), (yes_idx_2))

    return tf.case([(pred_fn_1, f1), (pred_fn_2, f2), (pred_fn_12, f12)])

@tf.function
def SE3_to_se3(T):
    """
    Take a SE(3) and convert it to se(3)

    Parameters
    ----------
    T (tf.Tensor):
        SE(3)
        N x 3 x 3

    Returns
    ------
    ret (tf.Tensor):
        se(3)
        N x 4 x 4
    """
    N = tf.shape(T)[0]
    R,p = SE3_to_Rp(T)
    b_1 = near_zero(tf.norm(R - tf.eye(3), axis=[-2,-1]))
    b_2 = tf.math.logical_not(b_1)
    idx_1 = tf.cast(tf.squeeze(tf.where(b_1), axis=1), tf.int32)
    idx_2 = tf.cast(tf.squeeze(tf.where(b_2), axis=1), tf.int32)

    def f1():
        first_three_row = tf.concat([tf.zeros((N,3,3)), tf.expand_dims(T[:,0:3,3], axis=2)], axis=2)
        last_row = tf.tile(tf.constant([[[0.,0.,0.,0.]]], tf.float32), tf.stack([N, 1, 1]))

        return tf.concat([first_three_row, last_row], axis=1)
    def f2():
        acosinput = (tf.linalg.trace(R) - 1) / 2.0
        acosinput = tf.clip_by_value(acosinput, clip_value_min=-1., clip_value_max=1.)
        theta = tf.math.acos(acosinput)
        omgmat = SO3_to_so3(R)
        a = tf.eye(3) - omgmat/2.0
        b = tf.expand_dims(tf.expand_dims((1.0/theta - 1.0/tf.tan(theta/2.0)/2.0)/theta, axis=1), axis=1) * tf.matmul(omgmat, omgmat)
        c = tf.matmul(a+b, tf.expand_dims(T[:,0:3,3], axis=2))

        return tf.concat([tf.concat([omgmat, c], axis=2), tf.tile(tf.constant([[[0., 0., 0., 0.]]], tf.float32), tf.stack([N,1,1]))], axis=1)
    def f12():
        T_1 = tf.gather(T, idx_1)
        first_three_row = tf.concat([tf.zeros((tf.shape(idx_1)[0],3,3)), tf.expand_dims(T_1[:,0:3,3], axis=2)], axis=2)
        last_row = tf.tile(tf.constant([[[0.,0.,0.,0.]]], tf.float32), tf.stack([tf.shape(idx_1)[0], 1, 1]))
        out_1 = tf.concat([first_three_row, last_row], axis=1)
        out_1 = tf.scatter_nd(tf.expand_dims(idx_1, axis=1), out_1, tf.stack([N, 4, 4]))

        T_2 = tf.gather(T, idx_2)
        R_2 = tf.gather(R, idx_2)
        acosinput = (tf.linalg.trace(R_2) - 1) / 2.0
        acosinput = tf.clip_by_value(acosinput, clip_value_min=-1., clip_value_max=1.)
        theta = tf.math.acos(acosinput)
        omgmat = SO3_to_so3(R_2)
        a = tf.eye(3) - omgmat/2.0
        b = tf.expand_dims(tf.expand_dims((1.0/theta - 1.0/tf.tan(theta/2.0)/2.0)/theta, axis=1), axis=1) * tf.matmul(omgmat, omgmat)
        c = tf.matmul(a+b, tf.expand_dims(T_2[:,0:3,3], axis=2))
        out_2 = tf.concat([tf.concat([omgmat, c], axis=2), tf.tile(tf.constant([[[0., 0., 0., 0.]]], tf.float32), tf.stack([tf.shape(idx_2)[0],1,1]))], axis=1)
        out_2 = tf.scatter_nd(tf.expand_dims(idx_2, axis=1), out_2, tf.stack([N, 4, 4]))

        return out_1 + out_2


    no_idx_1 = tf.math.equal(tf.shape(idx_1)[0], 0)
    yes_idx_1 = tf.math.logical_not(no_idx_1)
    no_idx_2 = tf.math.equal(tf.shape(idx_2)[0], 0)
    yes_idx_2 = tf.math.logical_not(no_idx_2)

    pred_fn_1 = tf.math.logical_and((yes_idx_1), (no_idx_2))
    pred_fn_2 = tf.math.logical_and((no_idx_1), (yes_idx_2))
    pred_fn_12 = tf.math.logical_and((yes_idx_1), (yes_idx_2))

    return tf.case([(pred_fn_1, f1), (pred_fn_2, f2), (pred_fn_12, f12)])

@tf.function
def fk_body(M, Blist, thetalist):
    """
    Forward Kinematics within Body Frame

    Parameters
    ----------
    M (tf.Tensor):
        SE(3) of the end-effector
        4 x 4
    Blist (list of tf.Tensor):
        List of the joint screw axes in the end-effector frame when the
        manipulator is at the home position
        [6, 6, ..., 6]
    thetalist (tf.Tensor):
        Joint positions
        N x nq

    Returns
    ------
    ret (tf.Tensor):
        SE(3) of the End-effector
        N x 4 x 4
    """
    N, nq = thetalist.shape
    T = tf.tile(tf.expand_dims(M, axis=0), tf.stack([N,1,1],0))
    for i in range(nq):
        T = tf.matmul(T, se3_to_SE3(vec_to_se3(Blist[i] * tf.expand_dims(thetalist[:,i], axis=1))))
    return T

@tf.function
def fk_space(M,Slist,thetalist):
    """
    Forward Kinematics within Space Frame

    Parameters
    ----------
    M (tf.Tensor):
        SE(3) of the end-effector
        4 x 4
    Slist (list of tf.Tensor):
        List of the joint screw axes in the space frame when the
        manipulator is at the home position
        [6, 6, ..., 6]
    thetalist (tf.Tensor):
        Joint positions
        N x nq

    Returns
    ------
    ret (tf.Tensor):
        SE(3) of the End-effector
    """
    N, nq = thetalist.shape
    T = tf.tile(tf.expand_dims(M, axis=0), tf.stack([N,1,1],0))
    for i in range(nq-1,-1,-1):
        T = tf.matmul(se3_to_SE3(vec_to_se3(Slist[i] * tf.expand_dims(thetalist[:,i],axis=1))),T)
    return T

@tf.function
def jac_body(Blist,thetalist):
    """
    Jacobian in the Body Frame

    Parameters
    ----------
    Blist (list of tf.Tensor):
        List of the joint screw axes in the end-effector frame when the
        manipulator is at the home position
        [6, 6, ..., 6]
    thetalist (tf.Tensor):
        Joint positions
        N x nq

    Returns
    ------
    ret (tf.Tensor):
        Body jacobian
        N x 6 x nv
    """
    N, nq = thetalist.shape
    Jb = [None] * nq
    for i in range(nq):
        Jb[i] = tf.tile(tf.expand_dims(Blist[i], axis=0), tf.constant([N,1]))
    T = tf.tile(tf.expand_dims(tf.eye(4), axis=0), tf.constant([N,1,1]))
    for i in range(nq-2,-1,-1):
        T = tf.matmul(T, se3_to_SE3(vec_to_se3(Blist[i+1] * -tf.expand_dims(thetalist[:,i+1],axis=1))))
        Jb[i] = tf.squeeze(tf.matmul(adjoint(T), tf.expand_dims(Blist[i], axis=1)), axis=2)
    return tf.stack(Jb, axis=2)

@tf.function
def jac_space(Slist, thetalist):
    """
    Jacobian in the Body Frame

    Parameters
    ----------
    Slist (list of tf.Tensor):
        List of the joint screw axes in the space frame when the
        manipulator is at the home position
        [6, 6, ..., 6]
    thetalist (tf.Tensor)
        Joint positions
        N x nq

    Returns
    ------
    ret (tf.Tensor):
        Space jacobian
        N x 6 x nv
    """
    N, nq = thetalist.shape
    Js = None * nq
    for i in range(nq):
        Js[i] = tf.tile(tf.expand_dims(S, axis=0), tf.constant([N,1]))
    T = tf.tile(tf.expand_dims(tf.eye(4), axis=0), tf.constant([N,1,1]))
    for i in range(1, nq):
        T = tf.matmul(T, se3_to_SE3(vec_to_se3(Slist[i-1] * tf.expand_dims(thetalist[:,i-1], axis=1))))
        Js[i] = tf.squeeze(tf.matmul(adjoint(T), tf.expand_dims(Slist[i], axis=1)), axis=2)
    return tf.stack(Js, axis=2)

@tf.function
def ik_body_single_(Blist, M, T, thetalist0, eomg, ev):
    """
    Inverse Kinematics within Body Frame

    Parameters
    ----------
    Blist (list of tf.Tensor):
        List of the joint screw axes in the end-effector frame when the
        manipulator is at the home position
        [6, 6, ..., 6]
    M (tf.Tensor):
        SE(3) of the end-effector
        4 x 4
    T (tf.Tensor):
        Desired end-effector SE(3)
        4 x 4
    thetalist0 (tf.Tensor)
        Initial guess of joint angles that are close to satisfying T
        nq
    eomg (float):
        A small positive tolerance on the end-effector orientation error.
    ev (float):
        A small positive tolerance on the end-effector linear position error.

    Returns
    ------
    thetalist (tf.Tensor):
        Joint angles that achieve T within the specified tolerances
        nq
    success (list of bool):
        A logical value where True means that the function found a solution
        and False means that it ran through the set number of maximum
        iterations without finding a solution within the tolerances eomg and ev.
    """
    thetalist = tf.identity(thetalist0)
    i = 0
    maxiterations = 20
    Vb = tf.squeeze(se3_to_vec(SE3_to_se3(tf.matmul(SE3_inv(fk_body(M,Blist,tf.expand_dims(thetalist, axis=0))),tf.expand_dims(T, axis=0)))), axis=0)
    err = tf.norm(Vb[0:3]) > eomg or tf.norm(Vb[3:6]) > ev
    while err and i < maxiterations:
        thetalist = thetalist + tf.tensordot(tf.linalg.pinv(tf.squeeze(jac_body(Blist, tf.expand_dims(thetalist, axis=0)), axis=0)),Vb,1)
        i = i + 1
        Vb = tf.squeeze(se3_to_vec(SE3_to_se3(tf.matmul(SE3_inv(fk_body(M,Blist, tf.expand_dims(thetalist, axis=0))),tf.expand_dims(T,axis=0)))), axis=0)
        err = tf.norm(Vb[0:3]) > eomg or tf.norm(Vb[3:6]) > ev
    return (thetalist, not err)

@tf.function
def ik_body(Blist, M, T, thetalist0, eomg, ev):
    """
    Inverse Kinematics within Body Frame

    Parameters
    ----------
    Blist (list of tf.Tensor):
        List of the joint screw axes in the end-effector frame when the
        manipulator is at the home position
        [6, 6, ..., 6]
    M (tf.Tensor):
        SE(3) of the end-effector
        4 x 4
    T (tf.Tensor):
        Desired end-effector SE(3)
        N x 4 x 4
    thetalist0 (tf.Tensor)
        Initial guess of joint angles that are close to satisfying T
        N x nq
    eomg (float):
        A small positive tolerance on the end-effector orientation error.
    ev (float):
        A small positive tolerance on the end-effector linear position error.

    Returns
    ------
    thetalist (tf.Tensor):
        Joint angles that achieve T within the specified tolerances
        nq
    success (list of bool):
        A logical value where True means that the function found a solution
        and False means that it ran through the set number of maximum
        iterations without finding a solution within the tolerances eomg and ev.
    """
    def ik_(T_th0_pair):
        des_T, th0 = T_th0_pair
        return ik_body_single_(Blist, M, des_T, th0, eomg, ev)
    return tf.map_fn(ik_, (T, thetalist0), dtype=(tf.float32, tf.bool))

@tf.function
def ik_space_single_(Slist,M,T,thetalist0,eomg,ev):
    """
    Inverse Kinematics within Space Frame

    Parameters
    ----------
    Slist (list of tf.Tensor):
        List of the joint screw axes in the space frame when the
        manipulator is at the home position
    M (tf.Tensor):
        SE(3) of the end-effector
        4 x 4
    T (tf.Tensor):
        Desired end-effector SE(3)
        4 x 4
    thetalist0 (tf.Tensor)
        Initial guess of joint angles that are close to satisfying T
        nq
    eomg (float):
        A small positive tolerance on the end-effector orientation error.
    ev (float):
        A small positive tolerance on the end-effector linear position error.

    Returns
    ------
    thetalist (list of float):
        Joint angles that achieve T within the specified tolerances
        nq
    success (bool):
        A logical value where True means that the function found a solution
        and False means that it ran through the set number of maximum
        iterations without finding a solution within the tolerances eomg and ev.
    """
    thetalist = tf.identity(thetalist0)
    i = 0
    maxiterations = 20
    Tsb = fk_space(M,Slist,tf.expand_dims(thetalist, axis=0))
    Vs = tf.tensordot(tf.squeeze(adjoint(Tsb), axis=0), tf.squeeze(se3_to_vec(SE3_to_se3(tf.matmul(SE3_inv(Tsb),tf.expand_dims(T,axis=0)))), axis=0), 1)
    err = tf.norm(Vs[0:3]) > eomg or tf.norm(Vs[3:6]) > ev
    while err and i < maxiterations:
        thetalist = thetalist + tf.tensordot(tf.linalg.pinv(tf.squeeze(jac_space(Slist, tf.expand_dims(thetalist, axis=0)), axis=0)),Vs,1)
        i = i + 1
        Tsb = fk_space(M,Slist,tf.expand_dims(thetalist, axis=0))
        Vs = tf.tensordot(tf.squeeze(adjoint(Tsb), axis=0), tf.squeeze(se3_to_vec(SE3_to_se3(tf.matmul(SE3_inv(Tsb),tf.expand_dims(T, axis=0)))),axis=0), 1)
        err = tf.norm(Vs[0:3]) > eomg or tf.norm(Vs[3:6]) > ev
    return (thetalist,not err)

@tf.function
def ik_space(Slist,M,T,thetalist0,eomg,ev):
    def ik_(T_th0_pair):
        des_T, th0 = T_th0_pair
        return ik_space_single_(Slist, M, des_T, th0, eomg, ev)
    return tf.map_fn(ik_, (T, thetalist0), dtype=(tf.float32, tf.bool))

@tf.function
def ad(V):
    """
    Takes 6-vector spatial velocity and returns the corresponding 6x6 matrix
    [adV]. Used to calculate the Lie bracket [V1, V2] = [adV1]V2.

    Parameters
    ----------
    V (tf.Tensor):
        Spatial velocity or twist
        N x 6

    Returns
    -------
    ret (tf.Tensor):
        Corresponding [adV]
        N x 6 x 6
    """

    N = V.shape[0]
    omgmat = vec_to_so3(V[:,0:3])
    linmat = vec_to_so3(V[:,3:6])
    return tf.concat([tf.concat([omgmat, tf.zeros((N,3,3))], axis=2), tf.concat([linmat, omgmat], axis=2)], axis=1)

@tf.function
def id_space(thetalist,dthetalist,ddthetalist,g,Ftip,Mlist,Glist,Slist):
    """
    Inverse dynamics

    Parameters
    ----------
    thetalist (tf.Tensor):
        Joint angles
        N x nq
    dthetalist (tf.Tensor):
        Joint velocities
        N x nq
    ddthetalist (tf.Tensor):
        Joint accelerations
        N x nq
    g (tf.Tensor):
        Gravity vector
        3
    Ftip (tf.Tensor):
        Spatial force applied by the end-effector expressed in frame {n+1}
        N x 6
    Mlist (list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
        [4 x 4, 4 x 4, ..., 4 x 4]
    Glist (list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
        [6 x 6, 6 x 6, ..., 6 x 6]
    Slist (list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.
        [6, 6, ..., 6]

    Returns
    -------
    taulist (tf.Tensor):
        Joint torques
        N x nq
    """
    N, nq = thetalist.shape
    Mi = tf.expand_dims(tf.eye(4), axis=0) # (1, 4, 4)
    Ai = [None]*nq # len([(1,6), (1,6), ..., (1,6)]) = nq
    AdTi = [None]*(nq+1) # lne([(N,6,6), (N,6,6), ..., (N,6,6)]) = nq+1
    Vi = [tf.zeros((N,6))]*(nq+1) # len([(N,6), (N,6), ..., (N,6)]) = nq+1
    Vdi = [None]*(nq+1) # len([(N,6), (N,6), ..., (N,6)]) = nq+1
    Vdi[0] = tf.tile(tf.expand_dims(tf.concat([tf.zeros(3), -g], axis=0), axis=0), tf.constant([N, 1]))
    AdTi[nq] = adjoint(SE3_inv(tf.expand_dims(Mlist[nq], axis=0)))
    Fi = tf.identity(Ftip) # (N,6)
    taulist = [None] * nq # len([(N), (N), ..., (N)]) = nq
    for i in range(nq):
        Mi = tf.matmul(Mi,tf.expand_dims(Mlist[i], axis=0)) # (1,4,4)
        Ai[i] = tf.squeeze(tf.matmul(adjoint(SE3_inv(Mi)), tf.expand_dims(Slist[i], axis=1)), axis=2) # (1, 6)
        AdTi[i] = adjoint(tf.matmul(se3_to_SE3(vec_to_se3(Ai[i] * -tf.expand_dims(thetalist[:,i], axis=1))), SE3_inv(tf.expand_dims(Mlist[i],axis=0)))) # (N, 6, 6)
        Vi[i+1] = tf.squeeze(tf.matmul(AdTi[i],tf.expand_dims(Vi[i],axis=2)),axis=2) + Ai[i] * tf.expand_dims(dthetalist[:,i],axis=1) # (N, 6)
        Vdi[i+1] = tf.squeeze(tf.matmul(AdTi[i],tf.expand_dims(Vdi[i],axis=2)),axis=2) + Ai[i] * tf.expand_dims(ddthetalist[:,i],axis=1) + tf.squeeze(tf.matmul(ad(Vi[i+1]),tf.expand_dims(Ai[i],axis=2)),axis=2) * tf.expand_dims(dthetalist[:,i],axis=1) # (N, 6)
    for i in range(nq-1,-1,-1):
        Fi = tf.squeeze(tf.matmul(tf.transpose(AdTi[i+1], perm=[0,2,1]),tf.expand_dims(Fi,axis=2)),axis=2) + tf.squeeze(tf.matmul(tf.expand_dims(Glist[i],axis=0),tf.expand_dims(Vdi[i+1],axis=2)),axis=2) - tf.squeeze(tf.matmul(tf.transpose(ad(Vi[i+1]),perm=[0,2,1]), tf.matmul(tf.expand_dims(Glist[i],axis=0),tf.expand_dims(Vi[i+1],axis=2))),axis=2) # (N, 6)
        taulist[i] = tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(Fi,axis=2),perm=[0,2,1]),tf.expand_dims(Ai[i],axis=2)),axis=[1,2]) # (N,1)
    return tf.stack(taulist, axis=1)

@tf.function
def mass_matrix(thetalist,Mlist,Glist,Slist):
    """
    Mass matrix

    Parameters
    ----------
    thetalist (tf.Tensor):
        Joint angles
        N x nq
    Mlist (list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
        [4 x 4, 4 x 4, ..., 4 x 4]
    Glist (list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
        [6 x 6, 6 x 6, ..., 6 x 6]
    Slist (list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.
        [6, 6, ..., 6]

    Returns
    -------
    M (tf.Tensor):
        Mass matrix
        N x nq x nq
    """
    N, nq = thetalist.shape
    M = [None]*nq # len([(N,nq), (N,nq), ..., (N,nq)] = nq
    for i in range(nq):
        ddthetalist = [0]*nq
        ddthetalist[i] = 1.
        ddth = tf.tile(tf.expand_dims(tf.constant(ddthetalist, tf.float32),axis=0), tf.constant([N,1],tf.int32))
        M[i] = id_space(thetalist, tf.zeros((N,nq)), ddth, tf.zeros(3), tf.zeros((N,6)), Mlist, Glist, Slist)
    return tf.stack(M, axis=2)

@tf.function
def coriolis_forces(thetalist,dthetalist,Mlist,Glist,Slist):
    """
    Coriolis

    Parameters
    ----------
    thetalist (tf.Tensor):
        Joint angles
        N x nq
    dthetalist (tf.Tensor):
        Joint velocities
        N x nq
    Mlist (list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
        [4 x 4, 4 x 4, ..., 4 x 4]
    Glist (list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
        [6 x 6, 6 x 6, ..., 6 x 6]
    Slist (list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.
        [6, 6, ..., 6]

    Returns
    -------
    b (tf.Tensor):
        Coriolis vector
        N x nq
    """
    N, nq = thetalist.shape
    return id_space(thetalist,dthetalist,tf.zeros((N,nq)),tf.zeros(3),tf.zeros((N,6)),Mlist,Glist,Slist)

@tf.function
def gravity_forces(thetalist,g,Mlist,Glist,Slist):
    """
    Gravity

    Parameters
    ----------
    thetalist (tf.Tensor):
        Joint angles
        N x nq
    Mlist (list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
        [4 x 4, 4 x 4, ..., 4 x 4]
    Glist (list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
        [6 x 6, 6 x 6, ..., 6 x 6]
    Slist (list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.
        [6, 6, ..., 6]

    Returns
    -------
    g (tf.Tensor):
        Gravity vector
        N x nq
    """
    N, nq = thetalist.shape
    return id_space(thetalist,tf.zeros((N,nq)),tf.zeros((N,nq)),g,tf.zeros((N,6)),Mlist,Glist,Slist)

@tf.function
def end_effector_forces(thetalist,Ftip,Mlist,Glist,Slist):
    """
    Gravity

    Parameters
    ----------
    thetalist (tf.Tensor):
        Joint angles
        N x nq
    Ftip (tf.Tensor):
        Spatial force applied by the end-effector expressed in frame {n+1}.
        N x 6
    Mlist (list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
        [4 x 4, 4 x 4, ..., 4 x 4]
    Glist (list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
        [6 x 6, 6 x 6, ..., 6 x 6]
    Slist (list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.
        [6, 6, ..., 6]

    Returns
    -------
    JTFtip (list of tf.Tensor):
        Joint forces
        N x nq
    """
    N, nq = thetalist.shape
    return id_space(thetalist,tf.zeros((N,nq)),tf.zeros((N,nq)),tf.zeros(3),Ftip,Mlist,Glist,Slist)

@tf.function
def fd_space(thetalist,dthetalist,taulist,g,Ftip,Mlist,Glist,Slist):
    """
    Forward dynamics

    Parameters
    ----------
    thetalist (tf.Tensor):
        Joint angles
        N x nq
    dthetalist (tf.Tensor):
        Joint velocities
        N x nq
    taulist (tf.Tensor):
        Joint torques
        N x nq
    g (tf.Tensor):
        Gravity vector
        3
    Ftip (tf.Tensor):
        Spatial force applied by the end-effector expressed in frame {n+1}
        N x 6
    Mlist (list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
        [4 x 4, 4 x 4, ..., 4 x 4]
    Glist (list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
        [6 x 6, 6 x 6, ..., 6 x 6]
    Slist (list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.
        [6, 6, ..., 6]

    Returns
    -------
    qddot (tf.Tensor):
        joint torques
        N x nq
    """
    N, nq = thetalist.shape
    return tf.squeeze(tf.matmul(tf.linalg.inv(mass_matrix(thetalist,Mlist,Glist,Slist)), tf.expand_dims(taulist - coriolis_forces(thetalist,dthetalist, Mlist,Glist,Slist) - gravity_forces(thetalist,g,Mlist,Glist,Slist) - end_effector_forces(thetalist,Ftip,Mlist, Glist,Slist),axis=2)), axis=2)

@tf.function
def euler_step(thetalist,dthetalist,ddthetalist,dt):
    """
    Euler Step

    Parameters
    ----------
    thetalist (tf.Tensor):
        Joint angles
        N x nq
    dthetalist (tf.Tensor):
        Joint velocities
        N x nq
    ddthetalist (tf.Tensor):
        Joint accelerations
        N x nq
    dt (float):
        Delta t

    Returns
    -------
    thetalistNext (tf.Tensor):
        Next joint angles
    dthetalistNext (tf.Tensor):
        Next joint velocities
    """
    return thetalist + dt * dthetalist, dthetalist + dt * ddthetalist
