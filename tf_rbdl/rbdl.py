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
    z (float):
        Scalar input

    Returns
    -------
    ret (bool):
        True if it is near zero
    """
    return abs(z) < 1e-3

@tf.function
def normalize_vector(V):
    """
    Normalize vector

    Parameters
    ----------
    V (np.ndarray or tf.Tensor):
        Vector input

    Returns
    -------
    ret (np.ndarray or tf.Tensor):
        Normalized vector
    """
    if type(V) is np.ndarray:
        return V / np.linalg.norm(V)
    else:
        return V / tf.norm(V)

@tf.function
def SO3_inv(R):
    """
    Inverse of SO(3)

    Parameters
    ----------
    R (np.ndarray or tf.Tensor):
        SO(3)

    Returns
    -------
    ret (np.ndarray or tf.Tensor):
        Inverse of the input SO(3)
    """
    if type(R) is np.ndarray:
        return R.T
    else:
        return tf.transpose(R)

@tf.function
def vec_to_so3(omg):
    """
    Convert angular velocity to so(3)

    Parameters
    ----------
    omg (np.ndarray or tf.Tensor):
        Angular velocity

    Returns
    -------
    ret (np.ndarray or tf.Tensor):
        so(3)
    """

    if type(omg) is np.ndarray:
        return np.array([[0., -omg[2] , omg[1]], [omg[2], 0, -omg[0]], [-omg[1], omg[0], 0.]])
    else:
        return tf.stack([tf.stack([0., -omg[2], omg[1]], axis=0), tf.stack([omg[2], 0., -omg[0]], axis=0), tf.stack([-omg[1], omg[0], 0.], axis=0)], axis=0)

@tf.function
def so3_to_vec(so3mat):
    """
    Convert so(3) to angular velocity

    Parameters
    ----------
    so3mat (np.ndarray or tf.Tensor):
        so(3)

    Returns
    -------
    ret (np.ndarray or tf.Tensor):
        Angular velocity
    """
    if type(so3mat) is np.ndarray:
        return np.array([so3mat[2,1], so3mat[0,2], so3mat[1,0]])
    else:
        ret = tf.stack([so3mat[2,1], so3mat[0,2], so3mat[1,0]])
        return ret

@tf.function
def angvel_to_axis_ang(expc3):
    """
    Convert a 3d-vector of exponential coordinates for rotation to an unit
    rotation axis omghat and the corresponding rotation angle theta.

    Parameters
    ----------
    expc3 (np.ndarray or tf.Tensor):
        exponential coordinates for rotation

    Returns
    -------
    ret (tuple):
        (axis, angle)
    """
    if type(expc3) is np.ndarray :
        return (normalize_vector(expc3), np.linalg.norm(expc3))
    else:
        return (normalize_vector(expc3), tf.norm(expc3))

@tf.function
def so3_to_SO3(so3mat):
    """
    Convert so(3) to SO(3)

    Parameters
    ----------
    so3mat (np.ndarray or tf.Tensor):
        so(3)

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        SO(3)
    """

    omgtheta = so3_to_vec(so3mat)
    if type(so3mat) is np.ndarray:
        if near_zero(np.linalg.norm(omgtheta)):
            return np.eye(3)
        else:
            theta = angvel_to_axis_ang(omgtheta)[1]
            omgmat = so3mat / theta
            return np.eye(3) + np.sin(theta) * omgmat + (1 - np.cos(theta)) * np.dot(omgmat,omgmat)
    else:
        if near_zero(tf.norm(omgtheta)):
            return tf.eye(3)
        else:
            theta = angvel_to_axis_ang(omgtheta)[1]
            omgmat = so3mat / theta
            return tf.eye(3) + tf.sin(theta) * omgmat + (1 - tf.cos(theta)) * tf.matmul(omgmat,omgmat)

@tf.function
def SO3_to_so3(R):
    """
    Convert SO(3) to so(3)

    Parameters
    ----------
    R (np.ndarray or tf.Tensor):
        SO(3)

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        so(3)
    """

    if type(R) is np.ndarray:
        if near_zero(np.linalg.norm(R - np.eye(3))):
            return np.zeros((3,3))
        elif near_zero(np.trace(R) + 1):
            if not near_zero(1 + R[2][2]):
                omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) * np.array([R[0][2], R[1][2], 1 + R[2][2]])
            elif not near_zero(1 + R[1][1]):
                omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) * np.array([R[0][1], 1 + R[1][1], R[2][1]])
            else:
                omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) * np.array([1 + R[0][0], R[1][0], R[2][0]])
            return vec_to_so3(np.pi*omg)
        else:
            acosinput = (np.trace(R) - 1) / 2.0
            if acosinput > 1:
                acosinput = 1
            elif acosinput < -1:
                acosinput = -1
            theta = np.arccos(acosinput)
            return theta / 2.0 / np.sin(theta) * (R - R.T)
    else:
        if near_zero(tf.norm(R - tf.eye(3))):
            return tf.zeros((3,3))
        elif near_zero(tf.linalg.trace(R) + 1):
            if not near_zero(1 + R[2][2]):
                omg = tf.stack([R[0,2], R[1,2], 1+R[2,2]], axis=0)
                omg = (1.0 / tf.sqrt(2 * (1 + R[2][2]))) * omg
            elif not near_zero(1 + R[1][1]):
                omg = tf.stack([R[0,1], 1+R[1,1], R[2,1]], axis=0)
                omg = (1.0 / tf.sqrt(2 * (1 + R[1][1]))) * omg
            else:
                omg = tf.stack([1+R[0,0], R[1,0], R[2,0]], axis=0)
                omg = (1.0 / tf.sqrt(2 * (1 + R[0][0]))) * omg
            return vec_to_so3(tf.constant(np.pi, tf.float32)*omg)
        else:
            acosinput = (tf.linalg.trace(R) - 1) / 2.0
            if acosinput > 1.:
                acosinput = 1.
            elif acosinput < -1.:
                acosinput = -1.
            theta = tf.math.acos(acosinput)
            return theta / 2.0 / tf.sin(theta) * (R - tf.transpose(R))

@tf.function
def Rp_to_SE3(R,p):
    """
    Convert R, p pair to SE3

    Parameters
    ----------
    R (np.ndarray or tf.Tensor):
        SO(3)
    p (np.ndarray or tf.Tensor):
        p

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        SE(3)
    """
    if type(R) is np.ndarray:
        ret = np.eye(4)
        ret[0:3, 0:3] = R
        ret[0:3, 3] = p
        return ret
    else:
        return  tf.concat([tf.concat([R, tf.expand_dims(p, axis=1)], axis=1), tf.constant([[0.,0.,0.,1.]], tf.float32)], axis=0)

@tf.function
def SE3_to_Rp(T):
    """
    Convert SE3 to R, p

    Parameters
    ----------
    T (np.ndarray or tf.Tensor):
        SE(3)

    Returns
    ------
    R (np.ndarray or tf.Tensor):
        SO(3)
    p (np.ndarray or tf.Tensor):
        translation
    """
    if type(T) is np.ndarray:
        R = T[0:3, 0:3]
        p = T[0:3, 3]
        return R, p

    else:
        return T[0:3,0:3], T[0:3,3]

@tf.function
def SE3_inv(T):
    """
    Inverse of SE3

    Parameters
    ----------
    T (np.ndarray or tf.Tensor):
        SE(3)

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        Inverse of the SE(3) input
    """
    R,p = SE3_to_Rp(T)
    if type(T) is np.ndarray:
        ret = np.eye(4)
        Rt = R.T
        ret[0:3, 0:3] = Rt
        ret[0:3, 3] = -np.dot(Rt, p)
        return ret
    else:
        Rt = tf.transpose(R)
        rotated_p = -tf.tensordot(Rt, p, 1)
        return Rp_to_SE3(Rt, rotated_p)

@tf.function
def vec_to_se3(V):
    """
    Take a 6d spatial velocity and convert it to se(3)

    Parameters
    ----------
    V (np.ndarray or tf.Tensor):
        spatial velocity (or twist)

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        se(3)
    """

    if type(V) is np.ndarray:
        ret = np.zeros((4,4))
        omg = V[0:3]
        so3 = vec_to_so3(omg)
        ret[0:3, 0:3] = so3
        ret[0:3, 3] = V[3:6]
        return ret
    else:
        so3 = vec_to_so3(V[0:3])
        return tf.concat([tf.concat([so3, tf.expand_dims(V[3:6], axis=1)], axis=1), tf.constant([[0., 0., 0., 0.]], tf.float32)], axis=0)

@tf.function
def se3_to_vec(se3mat):
    """
    Take a se(3) and convert it to 6d spatial velocity

    Parameters
    ----------
    ret (np.ndarray or tf.Tensor):
        se(3)

    Returns
    ------
    V (np.ndarray or tf.Tensor):
        spatial velocity (or twist)
    """
    if type(se3mat) is np.ndarray:
        return np.array([se3mat[2, 1], se3mat[0, 2], se3mat[1, 0], se3mat[0, 3],
            se3mat[1, 3], se3mat[2, 3]])
    else:
        return tf.stack([se3mat[2,1],se3mat[0,2],se3mat[1,0],se3mat[0,3],se3mat[1,3],se3mat[2,3]], axis=0)

@tf.function
def adjoint(T):
    """
    Take a SE(3) and compute corresponding adjoint operator

    Parameters
    ----------
    T (np.ndarray or tf.Tensor):
        SE(3)

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        Adjoint operator
    """
    R, p = SE3_to_Rp(T)
    if type(T) is np.ndarray:
        ret = np.zeros((6,6))
        ret[0:3, 0:3] = R
        ret[3:6, 0:3] = np.dot(vec_to_so3(p), R)
        ret[3:6, 3:6] = R
        return ret
    else:
        p_cross_R = tf.matmul(vec_to_so3(p), R)
        return tf.concat([tf.concat([R, tf.zeros((3,3))], axis=1), tf.concat([p_cross_R, R], axis=1)], axis=0)

@tf.function
def se3_to_SE3(se3mat):
    """
    Take a se(3) and convert it to SE(3)

    Parameters
    ----------
    se3mat (np.ndarray or tf.Tensor):
        se(3)

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        SE(3)
    """

    if type(se3mat) is np.ndarray:
        ret = np.eye(4)
        omgtheta = so3_to_vec(se3mat[0:3, 0:3])
        if near_zero(np.linalg.norm(omgtheta)):
            ret[0:3, 3] = se3mat[0:3, 3]
            return ret
        else:
            theta = angvel_to_axis_ang(omgtheta)[1]
            omgmat = se3mat[0:3, 0:3] / theta
            p = np.dot(np.eye(3) * theta + (1 - np.cos(theta)) * omgmat + (theta - np.sin(theta)) * np.dot(omgmat,omgmat), se3mat[0:3, 3]) / theta
            ret[0:3, 0:3] = so3_to_SO3(se3mat[0:3,0:3])
            ret[0:3, 3] = p
            return ret
    else:
        omgtheta = so3_to_vec(se3mat[0:3, 0:3])
        if near_zero(tf.norm(omgtheta)):
            return Rp_to_SE3(tf.eye(3), se3mat[0:3, 3])
        else:
            theta = angvel_to_axis_ang(omgtheta)[1]
            omgmat = se3mat[0:3, 0:3] / theta
            R = so3_to_SO3(se3mat[0:3, 0:3])
            p = tf.tensordot(tf.eye(3) * theta + (1 - tf.cos(theta)) * omgmat + (theta - tf.sin(theta)) * tf.matmul(omgmat,omgmat), se3mat[0:3, 3], 1) / theta
            return Rp_to_SE3(R, p)

@tf.function
def SE3_to_se3(T):
    """
    Take a SE(3) and convert it to se(3)

    Parameters
    ----------
    T (np.ndarray or tf.Tensor):
        SE(3)

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        se(3)
    """
    R,p = SE3_to_Rp(T)

    if type(T) is np.ndarray:
        ret = np.zeros((4,4))
        if near_zero(np.linalg.norm(R - np.eye(3))):
            ret[0:3, 3] = T[0:3, 3]
            return ret
        else:
            acosinput = (np.trace(R) - 1) / 2.0
            if acosinput > 1:
                acosinput = 1
            elif acosinput < -1:
                acosinput = -1
            theta = np.arccos(acosinput)
            omgmat = SO3_to_so3(R)
            ret[0:3, 0:3] = omgmat
            ret[0:3, 3] = np.dot(np.eye(3) - omgmat / 2.0 + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) * np.dot(omgmat,omgmat) / theta, T[0:3, 3])
            return ret
    else:
        if near_zero(tf.norm(R - tf.eye(3))):
            return tf.concat([tf.concat([tf.zeros((3,3)), tf.expand_dims(T[0:3,3], axis=1)], axis=1), tf.constant([[0., 0., 0., 0.]], tf.float32)], axis=0)
        else:
            acosinput = (tf.linalg.trace(R) - 1) / 2.0
            if acosinput > 1.:
                acosinput = 1.
            elif acosinput < -1.:
                acosinput = -1.
            theta = tf.math.acos(acosinput)
            omgmat = SO3_to_so3(R)
            return tf.concat([tf.concat([omgmat, tf.expand_dims(tf.tensordot(np.eye(3) - omgmat / 2.0 + (1.0 / theta - 1.0 / tf.tan(theta / 2.0) / 2) * tf.matmul(omgmat,omgmat) / theta, T[0:3, 3], 1), axis=1)], axis=1), tf.constant([[0., 0., 0., 0.]])], axis=0)

@tf.function
def fk_body(M, Blist, thetalist):
    """
    Forward Kinematics within Body Frame

    Parameters
    ----------
    M (np.ndarray or tf.Tensor):
        SE(3) of the end-effector
    Blist (list of np.ndarray or list of tf.Tensor):
        List of the joint screw axes in the end-effector frame when the
        manipulator is at the home position
    thetalist (np.ndarray or tf.Tensor):
        List of joint coordinates

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        SE(3) of the End-effector
    """
    if type(M) is np.ndarray:
        T = np.copy(M)
        for i in range(thetalist.shape[0]):
            T = np.copy(np.dot(T, se3_to_SE3(vec_to_se3(Blist[i] * thetalist[i]))))
        return T
    else:
        T = M
        for i in range(thetalist.shape[0]):
            T = tf.matmul(T, se3_to_SE3(vec_to_se3(Blist[i] * thetalist[i])))
        return T

@tf.function
def fk_space(M,Slist,thetalist):
    """
    Forward Kinematics within Space Frame

    Parameters
    ----------
    M (np.ndarray or tf.Tensor):
        SE(3) of the end-effector
    Slist (list of np.ndarray or list of tf.Tensor):
        List of the joint screw axes in the space frame when the
        manipulator is at the home position
    thetalist (np.ndarray or tf.Tensor):
        List of joint coordinates

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        SE(3) of the End-effector
    """
    if type(M) is np.ndarray:
        T = np.copy(M)
        for i in range(thetalist.shape[0]-1,-1,-1):
            T = np.copy(np.dot(se3_to_SE3(vec_to_se3(Slist[i] * thetalist[i])),T))
        return T
    else:
        T = M
        for i in range(thetalist.shape[0]-1,-1,-1):
            T = tf.matmul(se3_to_SE3(vec_to_se3(Slist[i] * thetalist[i])),T)
        return T

@tf.function
def jac_body(Blist,thetalist):
    """
    Jacobian in the Body Frame

    Parameters
    ----------
    Blist (list of np.ndarray or list of tf.Tensor):
        List of the joint screw axes in the end-effector frame when the
        manipulator is at the home position
    thetalist (np.ndarray or tf.Tensor):
        List of joint coordinates

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        6xn matrix
    """
    if type(Blist[0]) is np.ndarray:
        Jb = []
        for B in Blist:
            Jb.append(np.copy(B))
        T = np.eye(4)
        for i in range(thetalist.shape[0]-2,-1,-1):
            T = np.copy(np.dot(T, se3_to_SE3(vec_to_se3(Blist[i+1] * -thetalist[i+1]))))
            Jb[i] = np.dot(adjoint(T),Blist[i])
        return np.stack(Jb, axis=1)
    else:
        Jb = []
        for B in Blist:
            Jb.append(tf.identity(B))
        T = tf.eye(4)
        for i in range(thetalist.shape[0]-2,-1,-1):
            T = tf.matmul(T, se3_to_SE3(vec_to_se3(Blist[i+1] * -thetalist[i+1])))
            Jb[i] = tf.tensordot(adjoint(T),Blist[i], 1)
        return tf.stack(Jb, axis=1)

@tf.function
def jac_space(Slist, thetalist):
    """
    Jacobian in the Body Frame

    Parameters
    ----------
    Slist (list of np.ndarray or list of tf.Tensor):
        List of the joint screw axes in the space frame when the
        manipulator is at the home position
    thetalist (np.ndarray or tf.Tensor)
        List of joint coordinates

    Returns
    ------
    ret (np.ndarray or tf.Tensor):
        6xn matrix
    """
    if type(Slist[0]) is np.ndarray:
        Js = []
        for S in Slist:
            Js.append(np.copy(S))
        T = np.eye(4)
        for i in range(1,thetalist.shape[0]):
            T = np.copy(np.dot(T, se3_to_SE3(vec_to_se3(Slist[i-1] * thetalist[i-1]))))
            Js[i] = np.dot(adjoint(T),Slist[i])
        return np.stack(Js, axis=1)
    else:
        Js = []
        for S in Slist:
            Js.append(tf.identity(S))
        T = tf.eye(4)
        for i in range(1,thetalist.shape[0]):
            T = tf.matmul(T, se3_to_SE3(vec_to_se3(Slist[i-1] * thetalist[i-1])))
            Js[i] = tf.tensordot(adjoint(T),Slist[i], 1)
        return tf.stack(Js, axis=1)

@tf.function
def ik_body(Blist, M, T, thetalist0, eomg, ev):
    """
    Inverse Kinematics within Body Frame

    Parameters
    ----------
    Blist (list of np.ndarray or list of tf.Tensor):
        List of the joint screw axes in the end-effector frame when the
        manipulator is at the home position
    M (np.ndarray or tf.Tensor):
        SE(3) of the end-effector
    T (np.ndarray or tf.Tensor):
        Desired end-effector SE(3)
    thetalist0 (np.ndarray or tf.Tensor)
        Initial guess of joint angles that are close to satisfying T
    eomg (float):
        A small positive tolerance on the end-effector orientation error.
    ev (float):
        A small positive tolerance on the end-effector linear position error.

    Returns
    ------
    thetalist (list of float):
        Joint angles that achieve T within the specified tolerances
    success (bool):
        A logical value where True means that the function found a solution
        and False means that it ran through the set number of maximum
        iterations without finding a solution within the tolerances eomg and ev.
    """
    if type(T) is np.ndarray:
        thetalist = np.copy(thetalist0)
        i = 0
        maxiterations = 20
        Vb = se3_to_vec(SE3_to_se3(np.dot(SE3_inv(fk_body(M,Blist, thetalist)),T)))
        err = np.linalg.norm(Vb[0:3]) > eomg or np.linalg.norm(Vb[3:6]) > ev
        while err and i < maxiterations:
            thetalist = thetalist + np.dot(np.linalg.pinv(jac_body(Blist, thetalist)),Vb)
            i = i + 1
            Vb = se3_to_vec(SE3_to_se3(np.dot(SE3_inv(fk_body(M,Blist,thetalist)),T)))
            err = np.linalg.norm(Vb[0:3]) > eomg or np.linalg.norm(Vb[3:6]) > ev
        return (thetalist, not err)
    else:
        thetalist = tf.identity(thetalist0)
        i = 0
        maxiterations = 20
        Vb = se3_to_vec(SE3_to_se3(tf.matmul(SE3_inv(fk_body(M,Blist,thetalist)),T)))
        err = tf.norm(Vb[0:3]) > eomg or tf.norm(Vb[3:6]) > ev
        while err and i < maxiterations:
            thetalist = thetalist + tf.tensordot(tf.linalg.pinv(jac_body(Blist, thetalist)),Vb,1)
            i = i + 1
            Vb = se3_to_vec(SE3_to_se3(tf.matmul(SE3_inv(fk_body(M,Blist, thetalist)),T)))
            err = tf.norm(Vb[0:3]) > eomg or tf.norm(Vb[3:6]) > ev
        return (thetalist, not err)

@tf.function
def ik_space(Slist,M,T,thetalist0,eomg,ev):
    """
    Inverse Kinematics within Space Frame

    Parameters
    ----------
    Slist (list of np.ndarray or list of tf.Tensor):
        List of the joint screw axes in the space frame when the
        manipulator is at the home position
    M (np.ndarray or tf.Tensor):
        SE(3) of the end-effector
    T (np.ndarray or tf.Tensor):
        Desired end-effector SE(3)
    thetalist0 (np.ndarray or tf.Tensor)
        Initial guess of joint angles that are close to satisfying T
    eomg (float):
        A small positive tolerance on the end-effector orientation error.
    ev (float):
        A small positive tolerance on the end-effector linear position error.

    Returns
    ------
    thetalist (list of float):
        Joint angles that achieve T within the specified tolerances
    success (bool):
        A logical value where True means that the function found a solution
        and False means that it ran through the set number of maximum
        iterations without finding a solution within the tolerances eomg and ev.
    """
    if type(T) is np.ndarray:
        thetalist = np.copy(thetalist0)
        i = 0
        maxiterations = 20
        Tsb = fk_space(M,Slist,thetalist)
        Vs = np.dot(adjoint(Tsb), se3_to_vec(SE3_to_se3(np.dot(SE3_inv(Tsb),T))))
        err = np.linalg.norm(Vs[0:3]) > eomg or np.linalg.norm(Vs[3:6]) > ev
        while err and i < maxiterations:
            thetalist = thetalist + np.dot(np.linalg.pinv(jac_space(Slist, thetalist)),Vs)
            i = i + 1
            Tsb = fk_space(M,Slist,thetalist)
            Vs = np.dot(adjoint(Tsb), se3_to_vec(SE3_to_se3(np.dot(SE3_inv(Tsb),T))))
            err = np.linalg.norm(Vs[0:3]) > eomg or np.linalg.norm(Vs[3:6]) > ev
        return (thetalist,not err)
    else:
        thetalist = tf.identity(thetalist0)
        i = 0
        maxiterations = 20
        Tsb = fk_space(M,Slist,thetalist)
        Vs = tf.tensordot(adjoint(Tsb), se3_to_vec(SE3_to_se3(tf.matmul(SE3_inv(Tsb),T))), 1)
        err = tf.norm(Vs[0:3]) > eomg or tf.norm(Vs[3:6]) > ev
        while err and i < maxiterations:
            thetalist = thetalist + tf.tensordot(tf.linalg.pinv(jac_space(Slist, thetalist)),Vs,1)
            i = i + 1
            Tsb = fk_space(M,Slist,thetalist)
            Vs = tf.tensordot(adjoint(Tsb), se3_to_vec(SE3_to_se3(tf.matmul(SE3_inv(Tsb),T))), 1)
            err = tf.norm(Vs[0:3]) > eomg or tf.norm(Vs[3:6]) > ev
        return (thetalist,not err)

@tf.function
def ad(V):
    """
    Takes 6-vector spatial velocity and returns the corresponding 6x6 matrix
    [adV]. Used to calculate the Lie bracket [V1, V2] = [adV1]V2.

    Parameters
    ----------
    V (np.ndarray or tf.Tensor):
        Spatial velocity or twist

    Returns
    -------
    ret (np.ndarray or tf.Tensor):
        Corresponding [adV]
    """
    if type(V) is np.ndarray:
        ret = np.zeros((6,6))
        omgmat = vec_to_so3(np.array([V[0], V[1], V[2]]))
        linmat = vec_to_so3(np.array([V[3], V[4], V[5]]))
        ret[0:3, 0:3] = omgmat
        ret[3:6, 0:3] = linmat
        ret[3:6, 3:6] = omgmat
        return ret
    else:
        omgmat = vec_to_so3(V[0:3])
        linmat = vec_to_so3(V[3:6])
        return tf.concat([tf.concat([omgmat, tf.zeros((3,3))], axis=1), tf.concat([linmat, omgmat], axis=1)], axis=0)

@tf.function
def id_space(thetalist,dthetalist,ddthetalist,g,Ftip,Mlist,Glist,Slist):
    """
    Inverse dynamics

    Parameters
    ----------
    thetalist (np.ndarray or tf.Tensor):
        Joint angles
    dthetalist (np.ndarray or tf.Tensor):
        Joint velocities
    ddthetalist (np.ndarray or tf.Tensor):
        Joint accelerations
    g (np.ndarray or tf.Tensor):
        Gravity vector
    Ftip (np.ndarray or tf.Tensor):
        Spatial force applied by the end-effector expressed in frame {n+1}
    Mlist (list of np.array or list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
    Glist (list of np.array or list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
    Slist (list of np.array or list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.

    Returns
    -------
    taulist (np.array or list of tf.Tensor):
        joint torques
    """
    if type(g) is np.ndarray:
        n = thetalist.shape[0]
        Mi = np.eye(4)
        Ai = np.zeros((6,n))
        AdTi = [None]*(n + 1)
        Vi = np.zeros((6,n + 1))
        Vdi = np.zeros((6,n + 1))
        Vdi[3:6,0] = -g
        AdTi[n] = adjoint(SE3_inv(Mlist[n]))
        Fi = np.copy(Ftip)
        taulist = np.zeros(n)
        for i in range(n):
            Mi = np.dot(Mi,Mlist[i])
            Ai[:,i] = np.dot(adjoint(SE3_inv(Mi)),Slist[i])
            AdTi[i] = adjoint(np.dot(se3_to_SE3(vec_to_se3(Ai[:,i] * -thetalist[i])), SE3_inv(Mlist[i])))
            Vi[:,i + 1] = np.dot(AdTi[i],Vi[:,i]) + Ai[:,i] * dthetalist[i]
            Vdi[:,i + 1] = np.dot(AdTi[i],Vdi[:,i]) + Ai[:,i] * ddthetalist[i] + np.dot(ad(Vi[:,i + 1]),Ai[:,i]) * dthetalist[i]
        for i in range (n-1,-1,-1):
            Fi = np.dot(AdTi[i + 1].T,Fi) + np.dot(Glist[i],Vdi[:,i + 1]) - np.dot(ad(Vi[:,i + 1]).T, np.dot(Glist[i],Vi[:,i + 1]))
            taulist[i] = np.dot(Fi.T,Ai[:,i])
        return taulist
    else:
        n = thetalist.shape[0]
        Mi = tf.eye(4)
        Ai = [None]*n
        AdTi = [None]*(n+1)
        Vi = [tf.zeros(6)]*(n+1)
        Vdi = [None]*(n+1)
        Vdi[0] = tf.concat([tf.zeros(3), -g], axis=0)
        AdTi[n] = adjoint(SE3_inv(Mlist[n]))
        Fi = tf.identity(Ftip)
        taulist = [None] * n
        for i in range(n):
            Mi = tf.matmul(Mi,Mlist[i])
            Ai[i] = tf.tensordot(adjoint(SE3_inv(Mi)),Slist[i], 1)
            AdTi[i] = adjoint(tf.matmul(se3_to_SE3(vec_to_se3(Ai[i] * -thetalist[i])), SE3_inv(Mlist[i])))
            Vi[i+1] = tf.tensordot(AdTi[i],Vi[i], 1) + Ai[i] * dthetalist[i]
            Vdi[i+1] = tf.tensordot(AdTi[i],Vdi[i], 1) + Ai[i] * ddthetalist[i] + tf.tensordot(ad(Vi[i + 1]),Ai[i], 1) * dthetalist[i]
        for i in range(n-1,-1,-1):
            Fi = tf.tensordot(tf.transpose(AdTi[i+1]),Fi,1) + tf.tensordot(Glist[i],Vdi[i+1],1) - tf.tensordot(tf.transpose(ad(Vi[i+1])), tf.tensordot(Glist[i],Vi[i+1], 1), 1)
            taulist[i] = tf.tensordot(tf.transpose(Fi),Ai[i],1)
        return tf.stack(taulist, axis=0)

@tf.function
def mass_matrix(thetalist,Mlist,Glist,Slist):
    """
    Mass matrix

    Parameters
    ----------
    thetalist (np.ndarray or tf.Tensor):
        Joint angles
    Mlist (list of np.array or list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
    Glist (list of np.array or list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
    Slist (list of np.array or list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.

    Returns
    -------
    M (np.array or list of tf.Tensor):
        Mass matrix
    """
    if type(thetalist) is np.ndarray:
        n = thetalist.shape[0]
        M = np.zeros((n,n))
        for i in range (n):
            ddthetalist = np.zeros(n)
            ddthetalist[i] = 1.
            M[:,i] = id_space(thetalist, np.zeros(n), ddthetalist, np.zeros(3), np.zeros(6), Mlist, Glist, Slist)
        return M
    else:
        n = thetalist.shape[0]
        M = [None]*n
        for i in range(n):
            ddthetalist = [0]*n
            ddthetalist[i] = 1.
            M[i] = id_space(thetalist, tf.zeros(n), tf.constant(ddthetalist, tf.float32), tf.zeros(3), tf.zeros(6), Mlist, Glist, Slist)
        return tf.stack(M, axis=0)

@tf.function
def coriolis_forces(thetalist,dthetalist,Mlist,Glist,Slist):
    """
    Coriolis

    Parameters
    ----------
    thetalist (np.ndarray or tf.Tensor):
        Joint angles
    dthetalist (np.ndarray or tf.Tensor):
        Joint velocities
    Mlist (list of np.array or list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
    Glist (list of np.array or list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
    Slist (list of np.array or list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.

    Returns
    -------
    b (np.array or list of tf.Tensor):
        Coriolis vector
    """
    n = thetalist.shape[0]
    if type(thetalist) is np.ndarray:
        return id_space(thetalist,dthetalist, np.zeros(n), np.zeros(3), np.zeros(6),Mlist,Glist,Slist)
    else:
        return id_space(thetalist,dthetalist, tf.zeros(n), tf.zeros(3),tf.zeros(6),Mlist,Glist,Slist)

@tf.function
def gravity_forces(thetalist,g,Mlist,Glist,Slist):
    """
    Gravity

    Parameters
    ----------
    thetalist (np.ndarray or tf.Tensor):
        Joint angles
    Mlist (list of np.array or list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
    Glist (list of np.array or list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
    Slist (list of np.array or list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.

    Returns
    -------
    g (np.array or list of tf.Tensor):
        Gravity vector
    """
    n = thetalist.shape[0]
    if type(thetalist) is np.ndarray:
        return id_space(thetalist,np.zeros(n),np.zeros(n),g,np.zeros(6),Mlist,Glist,Slist)
    else:
        return id_space(thetalist,tf.zeros(n),tf.zeros(n),g,tf.zeros(6),Mlist,Glist,Slist)

@tf.function
def end_effector_forces(thetalist,Ftip,Mlist,Glist,Slist):
    """
    Gravity

    Parameters
    ----------
    thetalist (np.ndarray or tf.Tensor):
        Joint angles
    Ftip (np.ndarray or tf.Tensor):
        Spatial force applied by the end-effector expressed in frame {n+1}.
    Mlist (list of np.array or list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
    Glist (list of np.array or list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
    Slist (list of np.array or list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.

    Returns
    -------
    JTFtip (np.array or list of tf.Tensor):
        Joint forces
    """
    n = thetalist.shape[0]
    if type(thetalist) is np.ndarray:
        return id_space(thetalist,np.zeros(n),np.zeros(n),np.zeros(3),Ftip,Mlist,Glist,Slist)
    else:
        return id_space(thetalist,tf.zeros(n),tf.zeros(n),tf.zeros(3),Ftip,Mlist,Glist,Slist)

@tf.function
def fd_space(thetalist,dthetalist,taulist,g,Ftip,Mlist,Glist,Slist):
    """
    Forward dynamics

    Parameters
    ----------
    thetalist (np.ndarray or tf.Tensor):
        Joint angles
    dthetalist (np.ndarray or tf.Tensor):
        Joint velocities
    taulist (np.ndarray or tf.Tensor):
        Joint torques
    g (np.ndarray or tf.Tensor):
        Gravity vector
    Ftip (np.ndarray or tf.Tensor):
        Spatial force applied by the end-effector expressed in frame {n+1}
    Mlist (list of np.array or list of tf.Tensor):
        List of link frames {i} relative to {i-1} at the home position
    Glist (list of np.array or list of tf.Tensor):
        List of spatial inertia matrices Gi of the links.
    Slist (list of np.array or list of tf.Tensor):
        List of Screw axes Si of the joints in a space frame.

    Returns
    -------
    qddot (np.array or list of tf.Tensor):
        joint torques
    """
    if type(thetalist) is np.ndarray:
        return np.dot(np.linalg.inv(mass_matrix(thetalist,Mlist,Glist,Slist)), taulist - coriolis_forces(thetalist,dthetalist, Mlist,Glist,Slist) - gravity_forces(thetalist,g,Mlist,Glist,Slist) - end_effector_forces(thetalist,Ftip,Mlist, Glist,Slist))
    else:
        return tf.tensordot(tf.linalg.inv(mass_matrix(thetalist,Mlist,Glist,Slist)), taulist - coriolis_forces(thetalist,dthetalist, Mlist,Glist,Slist) - gravity_forces(thetalist,g,Mlist,Glist,Slist) - end_effector_forces(thetalist,Ftip,Mlist, Glist,Slist), 1)

@tf.function
def euler_step(thetalist,dthetalist,ddthetalist,dt):
    """
    Euler Step

    Parameters
    ----------
    thetalist (np.ndarray or tf.Tensor):
        Joint angles
    dthetalist (np.ndarray or tf.Tensor):
        Joint velocities
    ddthetalist (np.ndarray or tf.Tensor):
        Joint accelerations
    dt (float):
        Delta t

    Returns
    -------
    thetalistNext (np.array or list of tf.Tensor):
        Next joint angles
    dthetalistNext (np.array or list of tf.Tensor):
        Next joint velocities
    """
    return thetalist + dt * dthetalist, dthetalist + dt * ddthetalist
