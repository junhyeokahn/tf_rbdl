import tensorflow as tf

from tf_rbdl.liegroup import *
from tf_rbdl.kinematics import *

@tf.function
def id(theta,dtheta,ddtheta,g,pidlist,Mlist,Glist,Slist):
    """
    Inverse dynamics.
    Note that nbody == nq, since body only contains movable link.

    Parameters
    ----------
    theta (tf.Tensor):
        Joint angles
        (N,nq)
    dtheta (tf.Tensor):
        Joint velocities
        (N,nq)
    ddtheta (tf.Tensor):
        Joint accelerations
        (N,nq)
    g (tf.Tensor):
        Gravity vector
        (3,)
    pidlist (tf.Tensor):
        Parent body index.
        (nbody,)
    Mlist (tf.Tensor):
        Link frame i relative to p(i) at the home position
        (nbody,6,6)
    Glist (tf.Tensor):
        Spatial inertia matrices Gi of the links.
        (nbody,6,6)
    Slist (tf.Tensor):
        Screw axes Si of the joints in a space frame.
        (nq,6)

    Returns
    -------
    tau (tf.Tensor):
        Joint torques
        (N,nq)
    """
    N, nq = theta.shape
    nbody = Mlist.shape[0]

    Mi = tf.TensorArray(tf.float32, size=nbody, clear_after_read=False, colocate_with_first_write_call=True) # T_world_to_link_i : len([(4,4), (4,4), ..., (4,4)] = nbody
    Mi = Mi.write(0, tf.eye(4))
    Mi = Mi.write(0, Mlist[0])
    Ai = tf.TensorArray(tf.float32, size=nq, clear_after_read=False) # Twist of joint i to the link : len([(1,6), (1,6), ..., (1,6)]) = nq
    AdTi = tf.TensorArray(tf.float32, size=nbody, clear_after_read=False) # Adjoint_i_p(i) : len([(N,6,6), (N,6,6), ..., (N,6,6)]) = nbody
    Vi = tf.TensorArray(tf.float32, size=nbody+1, clear_after_read=False, colocate_with_first_write_call=True) # len([(N,6), (N,6), ..., (N,6)]) = nbody+1, Vi[0] <-- World, Vi[1] <-- Link 0
    Vi = Vi.write(0, tf.zeros((N, 6)))
    Vdi = tf.TensorArray(tf.float32, size=nbody+1, clear_after_read=False) # len([(N,6), (N,6), ..., (N,6)]) = nbody+1, This starts from the world.
    Vdi = Vdi.write(0, tf.tile(tf.expand_dims(tf.concat([tf.zeros(3), -g], axis=0), axis=0), tf.constant([N, 1])))
    Fi = tf.TensorArray(tf.float32, size=nbody, clear_after_read=False, colocate_with_first_write_call=True)
    Fi = Fi.write(0, tf.zeros((N,6))) # len([(N,6), (N,6), ..., (N,6)]) = nbody
    tau = tf.TensorArray(tf.float32, size=nq, clear_after_read=False) # len([(N), (N), ..., (N)]) = nq
    for i in tf.range(1,nbody):
        Mi = Mi.write(i, tf.matmul(Mi.read(pidlist[i]),Mlist[i])) # (1,4,4)
    for i in tf.range(nbody):
        Ai = Ai.write(i, tf.squeeze(tf.matmul(adjoint(SE3_inv(tf.expand_dims(Mi.read(i),0))), tf.expand_dims(Slist[i], axis=1)), axis=2)) # (1, 6)
        AdTi = AdTi.write(i, adjoint(tf.matmul(se3_to_SE3(vec_to_se3(Ai.read(i) * -tf.expand_dims(theta[:,i], axis=1))), SE3_inv(tf.expand_dims(Mlist[i],axis=0))))) # (N, 6, 6)
        Vi = Vi.write(i+1, tf.squeeze(tf.matmul(AdTi.read(i),tf.expand_dims(Vi.read(pidlist[i]+1),axis=2)),axis=2) + Ai.read(i) * tf.expand_dims(dtheta[:,i],axis=1)) # (N, 6)
        Vdi = Vdi.write(i+1, tf.squeeze(tf.matmul(AdTi.read(i),tf.expand_dims(Vdi.read(pidlist[i]+1),axis=2)),axis=2) + Ai.read(i) * tf.expand_dims(ddtheta[:,i],axis=1) + tf.squeeze(tf.matmul(ad(Vi.read(i+1)),tf.expand_dims(Ai.read(i),axis=2)),axis=2) * tf.expand_dims(dtheta[:,i],axis=1)) # (N, 6)
        Fi = Fi.write(i, tf.squeeze(tf.matmul(tf.expand_dims(Glist[i],axis=0),tf.expand_dims(Vdi.read(i+1),axis=2)),axis=2) - tf.squeeze(tf.matmul(tf.transpose(ad(Vi.read(i+1)),perm=[0,2,1]), tf.matmul(tf.expand_dims(Glist[i],axis=0),tf.expand_dims(Vi.read(i+1),axis=2))),axis=2)) # (N,6)
    for i in tf.range(nbody-1,0,-1):
        Fi = Fi.write(pidlist[i], Fi.read(pidlist[i]) + tf.squeeze(tf.matmul(tf.transpose(AdTi.read(i), perm=[0,2,1]),tf.expand_dims(Fi.read(i),axis=2)),axis=2))
    for i in tf.range(nq):
        tau = tau.write(i, tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(Fi.read(i),axis=2),perm=[0,2,1]),tf.expand_dims(Ai.read(i),axis=2)),axis=[1,2])) # (N,)
    return tf.transpose(tau.stack())

@tf.function
def mass_matrix(theta,pidlist,Mlist,Glist,Slist):
    """
    Mass matrix

    Parameters
    ----------
    theta (tf.Tensor):
        Joint angles
        N x nq
    pidlist (tf.Tensor):
        Parent body index.
        (nbody,)
    Mlist (tf.Tensor):
        Link frame i relative to p(i) at the home position
        (nbody,6,6)
    Glist (tf.Tensor):
        Spatial inertia matrices Gi of the links.
        (nbody,6,6)
    Slist (tf.Tensor):
        Screw axes Si of the joints in a space frame.
        (nq,6)

    Returns
    -------
    M (tf.Tensor):
        Mass matrix
        (N,nq,nq)
    """
    N, nq = theta.shape
    M = tf.TensorArray(tf.float32, size=nq, clear_after_read=False, colocate_with_first_write_call=True)
    M = M.write(0,tf.zeros((N,nq)))
    for i in tf.range(nq):
        ddtheta = tf.TensorArray(tf.float32, size=nq, clear_after_read=False, colocate_with_first_write_call=True)
        ddtheta = ddtheta.write(0,tf.zeros(N))
        ddtheta = ddtheta.write(i,tf.ones(N))
        M = M.write(i, id(theta,tf.zeros((N,nq)),tf.transpose(ddtheta.stack()),tf.zeros(3),pidlist,Mlist,Glist,Slist))
    return tf.transpose(M.stack(), perm=[1,0,2])

@tf.function
def coriolis_forces(theta,dtheta,pidlist,Mlist,Glist,Slist):
    """
    Coriolis

    Parameters
    ----------
    theta (tf.Tensor):
        Joint angles
        (N,nq)
    dtheta (tf.Tensor):
        Joint velocities
        (N,nq)
    pidlist (tf.Tensor):
        Parent body index.
        (nbody,)
    Mlist (tf.Tensor):
        Link frame i relative to p(i) at the home position
        (nbody,6,6)
    Glist (tf.Tensor):
        Spatial inertia matrices Gi of the links.
        (nbody,6,6)
    Slist (tf.Tensor):
        Screw axes Si of the joints in a space frame.
        (nq,6)

    Returns
    -------
    b (tf.Tensor):
        Coriolis vector
        (N,nq)
    """
    N, nq = theta.shape
    return id(theta,dtheta,tf.zeros((N,nq)),tf.zeros(3),pidlist,Mlist,Glist,Slist)

@tf.function
def gravity_forces(theta,g,pidlist,Mlist,Glist,Slist):
    """
    Gravity

    Parameters
    ----------
    theta (tf.Tensor):
        Joint angles
        (N,nq)
    pidlist (tf.Tensor):
        Parent body index.
        (nbody,)
    Mlist (tf.Tensor):
        Link frame i relative to p(i) at the home position
        (nbody,6,6)
    Glist (tf.Tensor):
        Spatial inertia matrices Gi of the links.
        (nbody,6,6)
    Slist (tf.Tensor):
        Screw axes Si of the joints in a space frame.
        (nq,6)

    Returns
    -------
    g (tf.Tensor):
        Gravity vector
        N x nq
    """
    N, nq = theta.shape
    return id(theta,tf.zeros((N,nq)),tf.zeros((N,nq)),g,pidlist,Mlist,Glist,Slist)

@tf.function
def fd(theta,dtheta,tau,g,pidlist,Mlist,Glist,Slist):
    """
    Forward dynamics

    Parameters
    ----------
    theta (tf.Tensor):
        Joint angles
        (N,nq)
    dtheta (tf.Tensor):
        Joint velocities
        (N,nq)
    pidlist (tf.Tensor):
        Parent body index.
        (nbody,)
    Mlist (tf.Tensor):
        Link frame i relative to p(i) at the home position
        (nbody,6,6)
    Glist (tf.Tensor):
        Spatial inertia matrices Gi of the links.
        (nbody,6,6)
    Slist (tf.Tensor):
        Screw axes Si of the joints in a space frame.
        (nq,6)

    Returns
    -------
    b (tf.Tensor):
        Coriolis vector
        (N,nq)
    """
    N, nq = theta.shape
    return tf.squeeze(tf.matmul(tf.linalg.inv(mass_matrix(theta,pidlist,Mlist,Glist,Slist)), tf.expand_dims(tau - coriolis_forces(theta,dtheta,pidlist,Mlist,Glist,Slist) - gravity_forces(theta,g,pidlist,Mlist,Glist,Slist),axis=2)), axis=2)

@tf.function
def euler_step(theta,dtheta,ddtheta,dt):
    """
    Euler Step

    Parameters
    ----------
    theta (tf.Tensor):
        Joint angles
        (N,nq)
    dtheta (tf.Tensor):
        Joint velocities
        (N,nq)
    ddtheta (tf.Tensor):
        Joint accelerations
        (N,nq)
    dt (float):
        Delta t

    Returns
    -------
    (thetalistNext, dthetalistNext) (tupe of tf.Tensor):
        Next joint angles and velocities
        (N,nq), (N,nq)
    """
    return theta + dt * dtheta, dtheta + dt * ddtheta
