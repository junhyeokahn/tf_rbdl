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
def so3_to_SO3_(so3mat):
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
    omgtheta = so3_to_vec(so3mat)
    c_1 = near_zero(tf.norm(omgtheta,axis=1))
    c_2 = tf.math.logical_not(c_1)
    b_1 = tf.cast(c_1, tf.int32)
    b_2 = tf.cast(c_2, tf.int32)
    idx_1 = tf.cast(tf.squeeze(tf.where(b_1), axis=1), tf.int32)
    idx_2 = tf.cast(tf.squeeze(tf.where(b_2), axis=1), tf.int32)

    partitions = b_1*0 + b_2*1
    partitioned_inp = tf.dynamic_partition(so3mat, partitions, 2)
    inp_1 = partitioned_inp[0]
    inp_2 = partitioned_inp[1]

    ret_1 = tf.tile( tf.expand_dims(tf.eye(3), axis=0), tf.stack([tf.shape(idx_1)[0], 1, 1], 0))

    omgtheta_2 = so3_to_vec(inp_2)
    theta_2 = tf.expand_dims(angvel_to_axis_ang(omgtheta_2)[1], axis=1)
    omgmat_2 = inp_2 / theta_2
    ret_2 = tf.eye(3) + tf.sin(theta_2) * omgmat_2 + (1 - tf.cos(theta_2)) * tf.matmul(omgmat_2,omgmat_2)

    rets = [ret_1,ret_2]
    ids = [idx_1,idx_2]
    return tf.dynamic_stitch(ids,rets)

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
    c_1 = near_zero(tf.norm(R - tf.eye(3), axis=[-2,-1]))
    c_2 = tf.math.logical_and(tf.math.logical_not(c_1), near_zero(tf.linalg.trace(R) + 1))
    c_21 = tf.math.logical_and(c_2, tf.math.logical_not(near_zero(1 + R[:,2,2])))
    c_22 = tf.math.logical_and(tf.math.logical_and(c_2, near_zero(1+R[:,2,2])), tf.math.logical_not(near_zero(1 + R[:,1,1])))
    c_23 = tf.math.logical_and(tf.math.logical_and(c_2, near_zero(1+R[:,2,2])), near_zero(1+R[:,1,1]))
    c_3 = tf.math.logical_and(tf.math.logical_not(c_1), tf.math.logical_not(c_2))
    b_1 = tf.cast((c_1), tf.int32)
    b_21 = tf.cast((c_21), tf.int32)
    b_22 = tf.cast((c_22), tf.int32)
    b_23 = tf.cast((c_23), tf.int32)
    b_3 = tf.cast((c_3), tf.int32)
    idx_1 = tf.cast(tf.squeeze(tf.where(b_1), axis=1), tf.int32)
    idx_21 = tf.cast(tf.squeeze(tf.where(b_21), axis=1), tf.int32)
    idx_22 = tf.cast(tf.squeeze(tf.where(b_22), axis=1), tf.int32)
    idx_23 = tf.cast(tf.squeeze(tf.where(b_23), axis=1), tf.int32)
    idx_3 = tf.cast(tf.squeeze(tf.where(b_3), axis=1), tf.int32)

    partitions = b_1*0 + b_21*1 + b_22*2 + b_23*3 + b_3*4
    partitioned_R = tf.dynamic_partition(R, partitions, 5)
    R_1 = partitioned_R[0]
    R_21 = partitioned_R[1]
    R_22 = partitioned_R[2]
    R_23 = partitioned_R[3]
    R_3 = partitioned_R[4]

    ret_1 = tf.zeros((tf.shape(R_1)[0], 3, 3))

    omg_21 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_21[:,2,2]))), axis=1) * tf.stack([R_21[:,0,2], R_21[:,1,2], 1+R_21[:,2,2]], axis=1)
    ret_21 = vec_to_so3(np.pi*omg_21)
    omg_22 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_22[:,1,1]))), axis=1) * tf.stack([R_22[:,0,1], 1+R_22[:,1,1], R_22[:,2,1]], axis=1)
    ret_22 = vec_to_so3(np.pi*omg_22)
    omg_23 = tf.expand_dims((1.0 / tf.sqrt(2 * (1 + R_23[:,0,0]))), axis=1) * tf.stack([1+R_23[:,0,0], R_23[:,1,0], R_23[:,2,0]], axis=1)
    ret_23 = vec_to_so3(np.pi*omg_23)

    acosinput = (tf.linalg.trace(R_3) - 1) / 2.0
    acosinput = tf.clip_by_value(acosinput, clip_value_min=-1., clip_value_max=1.)
    theta = tf.math.acos(acosinput)
    ret_3 = tf.expand_dims(tf.expand_dims(theta / 2.0 / tf.sin(theta), axis=1), axis=1) * (R_3 - tf.transpose(R_3, perm=[0,2,1]))

    rets = [ret_1,ret_21,ret_22,ret_23,ret_3]
    ids = [idx_1,idx_21,idx_22,idx_23,idx_3]
    return tf.dynamic_stitch(ids,rets)

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
    omgtheta = so3_to_vec(se3mat[:,0:3,0:3])
    c_1 = near_zero(tf.norm(omgtheta, axis=1))
    c_2 = tf.math.logical_not(c_1)
    b_1 = tf.cast(c_1, tf.int32)
    b_2 = tf.cast(c_2, tf.int32)
    idx_1 = tf.cast(tf.squeeze(tf.where(b_1), axis=1), tf.int32)
    idx_2 = tf.cast(tf.squeeze(tf.where(b_2), axis=1), tf.int32)

    partitions = b_1*0 + b_2*1
    partitioned_inp = tf.dynamic_partition(se3mat, partitions, 2)
    inp_1 = partitioned_inp[0]
    inp_2 = partitioned_inp[1]

    ret_1 = Rp_to_SE3( tf.tile(tf.expand_dims(tf.eye(3), axis=0), tf.stack([tf.shape(idx_1)[0], 1, 1])), inp_1[:,0:3,3] )

    omgtheta_2 = so3_to_vec(inp_2)
    theta_2 = angvel_to_axis_ang(omgtheta_2)[1]
    omgmat_2 = inp_2[:,0:3,0:3] / tf.expand_dims(theta_2, axis=1)
    R_2 = so3_to_SO3(inp_2[:,0:3,0:3])
    a_2 = tf.tile(tf.expand_dims(tf.eye(3), axis=0), tf.stack([tf.shape(inp_2)[0], 1, 1])) * tf.expand_dims(theta_2, axis=1)
    b_2 = tf.expand_dims((1-tf.cos(theta_2)), axis=1) * omgmat_2
    c_2 = tf.expand_dims(theta_2 - tf.sin(theta_2), axis=1) * tf.matmul(omgmat_2,omgmat_2)
    d_2 = a_2+b_2+c_2
    p_2 = tf.squeeze(tf.matmul(d_2, tf.expand_dims(inp_2[:,0:3,3], axis=2)) / tf.expand_dims(theta_2, axis=1), axis=2)
    ret_2 = Rp_to_SE3(R_2, p_2)

    rets = [ret_1,ret_2]
    ids = [idx_1,idx_2]
    return tf.dynamic_stitch(ids,rets)

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
    R, p = SE3_to_Rp(T)
    c_1 = near_zero(tf.norm(R - tf.eye(3), axis=[-2,-1]))
    c_2 = tf.math.logical_not(c_1)
    b_1 = tf.cast(c_1, tf.int32)
    b_2 = tf.cast(c_2, tf.int32)
    idx_1 = tf.cast(tf.squeeze(tf.where(b_1), axis=1), tf.int32)
    idx_2 = tf.cast(tf.squeeze(tf.where(b_2), axis=1), tf.int32)

    partitions = b_1*0 + b_2*1
    partitioned_T = tf.dynamic_partition(T, partitions, 2)
    partitioned_R = tf.dynamic_partition(R, partitions, 2)
    T_1 = partitioned_T[0]
    R_1 = partitioned_R[0]
    T_2 = partitioned_T[1]
    R_2 = partitioned_R[1]

    first_three_row = tf.concat([tf.zeros((tf.shape(idx_1)[0],3,3)), tf.expand_dims(T_1[:,0:3,3], axis=2)], axis=2)
    last_row = tf.tile(tf.constant([[[0.,0.,0.,0.]]], tf.float32), tf.stack([tf.shape(idx_1)[0], 1, 1]))
    ret_1 = tf.concat([first_three_row, last_row], axis=1)

    acosinput = (tf.linalg.trace(R_2) - 1) / 2.0
    acosinput = tf.clip_by_value(acosinput, clip_value_min=-1., clip_value_max=1.)
    theta = tf.math.acos(acosinput)
    omgmat = SO3_to_so3(R_2)
    a = tf.eye(3) - omgmat/2.0
    b = tf.expand_dims(tf.expand_dims((1.0/theta - 1.0/tf.tan(theta/2.0)/2.0)/theta, axis=1), axis=1) * tf.matmul(omgmat, omgmat)
    c = tf.matmul(a+b, tf.expand_dims(T_2[:,0:3,3], axis=2))
    ret_2 = tf.concat([tf.concat([omgmat, c], axis=2), tf.tile(tf.constant([[[0., 0., 0., 0.]]], tf.float32), tf.stack([tf.shape(idx_2)[0],1,1]))], axis=1)

    rets = [ret_1,ret_2]
    ids = [idx_1,idx_2]
    return tf.dynamic_stitch(ids,rets)

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
    Js = [None] * nq
    for i in range(nq):
        Js[i] = tf.tile(tf.expand_dims(Slist[i], axis=0), tf.constant([N,1]))
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

    N = tf.shape(V)[0]
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
