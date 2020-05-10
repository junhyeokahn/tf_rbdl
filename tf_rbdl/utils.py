import tensorflow as tf
import numpy as np
import os
import sys

from mujoco_py import load_model_from_path, MjSim, functions, MjViewer

from tf_rbdl.liegroup import *

def pretty_print_dictionary(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_print_dictionary(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

def quat_to_SO3(quat):
    quat = np.squeeze(np.asarray(quat))
    w, x, y, z = quat
    return np.matrix([
            [1 - 2*y*y-2*z*z, 2*x*y - 2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])

def initial_config_from_mjcf(file, ee_list, verbose=False):
    """
    Parameters
    ----------
    file (str):
        mjcf file
    ee_list (list of str):
        End-effector body names
    Returns
    -------
    ret (dict):
        dt (tf.Tensor)
        joint_armature (tf.Tensor):
            (nq,)
        actuator_gear (tf.Tensor):
            (nu,)
        g (tf.Tensor): Gravity
        Mlist (tf.Tensor):
            Link frame i relative to p(i) at the home position
            (nbody,6,6)
        Slist (tf.Tensor):
            Screw axes Si of the joints in a space frame.
            (nq,6)
        init_ee_SE3 (dict of tf.Tensor):
            Initial end-effector SE3
        Blist (dict of tf.Tensor):
            Joint screw axes in the end-effector frame when the
            manipulator is at the home position
            (nq_branch,6)
        Bidlist (dict of tf.Tensor):
            List of the jonit index related to the end-effector branch
            (nq_branch,)
        init_ee_SE3 (dict of tf.Tensor):
            4x4
        Glist (tf.Tensor):
            Spatial inertia matrices Gi of the links.
            (nbody,6,6)
        # j_st_id (tf.Tensor):
            # Start joint idx for each link. It handles multiple joints in one
            # body. Last element is nq.
            # (nbody+1,)
        # b_id (tf.Tensor):
            # Body idx for each joint.
            # (nq,)
        pidlist (tf.Tensor):
            Parent body index.
            (nbody,)

    """
    ret = dict()
    model = load_model_from_path(file)
    sim = MjSim(model)
    sim_state = sim.get_state()
    for i in range(sim.model.nq):
        sim_state.qpos[i] = 0.
        sim_state.qvel[i] = 0.
    sim.set_state(sim_state)
    sim.forward()

    nq, nv, na = sim.model.nq, sim.model.nv, sim.model.nu
    n_ee = len(ee_list)
    njoint, nbody = sim.model.njnt, sim.model.nbody-n_ee-1

    assert sim.model.nq + 1 + n_ee == sim.model.nbody
    assert nq == nv
    assert nq == njoint
    assert nq == nbody

    muj_world_id = 0
    muj_ee_id, muj_body_id = [], []
    muj_global_body_pos, muj_global_body_SO3 = np.zeros((sim.model.nbody, 3)), np.zeros((sim.model.nbody, 3, 3))
    muj_global_joint_pos, muj_global_joint_SO3 = np.zeros((sim.model.nq, 3)), np.zeros((sim.model.nq, 3, 3))
    muj_global_body_SO3[0] = np.eye(3)
    muj_global_inertial_pos, muj_global_inertial_SO3 = np.zeros((sim.model.nbody, 3)), np.zeros((sim.model.nbody, 3, 3))
    for i in range(sim.model.nbody):
        if sim.model.body_names[i] == "world":
            muj_world_id = i
        elif sim.model.body_names[i] in ee_list:
            muj_ee_id.append(sim.model.body_name2id(sim.model.body_names[i]))
        else:
            muj_body_id.append(i)
        muj_global_body_SO3[i] = sim.data.get_body_xmat(sim.model.body_names[i])
        muj_global_body_pos[i] = sim.data.get_body_xpos(sim.model.body_names[i])
        muj_global_inertial_pos[i] = muj_global_body_pos[i] + sim.model.body_ipos[i]
        muj_global_inertial_SO3[i] = np.dot(muj_global_body_SO3[i], quat_to_SO3(sim.model.body_iquat[i]))
    for i in range(sim.model.nq):
        body_id = sim.model.jnt_bodyid[i]
        muj_global_joint_SO3[i] = muj_global_body_SO3[body_id]
        muj_global_joint_pos[i] = muj_global_body_pos[body_id] + np.dot(muj_global_body_SO3[body_id], sim.model.jnt_pos[i])
    Bidlist = dict()
    for ee in ee_list:
        idlist_reverse = []
        ee_parent = sim.model.body_parentid[sim.model.body_name2id(ee)]
        while True:
            j_st_id = sim.model.body_jntadr[ee_parent]
            nj = sim.model.body_jntnum[ee_parent]
            idlist_reverse.extend([*range(j_st_id, j_st_id+nj)][::-1])
            ee_parent = sim.model.body_parentid[ee_parent]
            if (sim.model.body_names[ee_parent]=='world'):
                break
        Bidlist[ee] = idlist_reverse[::-1]

    if verbose:
        print("="*80)
        print("Infos aboue Mujoco Model")
        print("-"*80)
        print("Global Body Pos")
        print(muj_global_body_pos)
        print("-"*80)
        print("Global Body Ori")
        print(muj_global_body_SO3)
        print("-"*80)
        print("Global Inertia Pos")
        print(muj_global_inertial_pos)
        print("-"*80)
        print("Global Inertia Ori")
        print(muj_global_inertial_SO3)
        print("-"*80)
        print("Global Joint Pos")
        print(muj_global_joint_pos)
        print("-"*80)
        print("Global Joint Ori")
        print(muj_global_joint_SO3)
        for ee in ee_list:
            print("-"*80)
            print("{} Related Joint Id".format(ee))
            print(Bidlist[ee])
        print("="*80)

    ret['dt'] = tf.convert_to_tensor(sim.model.opt.timestep, tf.float32)
    ret['joint_armature'] = tf.convert_to_tensor(sim.model.dof_armature, tf.float32)
    if na is not 0:
        ret['actuator_gear'] = tf.convert_to_tensor(sim.model.actuator_gear[:,0], tf.float32)
    ret['g'] = tf.convert_to_tensor(sim.model.opt.gravity, tf.float32)
    ret['pidlist'] = tf.convert_to_tensor(sim.model.body_parentid[muj_body_id]-1, tf.int32)
    # ret['j_st_id'] = [None]*(nbody+1)
    # ret['j_st_id'][nbody] = nq
    ret['Mlist'] = [None]*nbody
    ret['Glist'] = [None]*nbody
    for i in range(nbody):
        muj_id = muj_body_id[i]
        muj_pid = sim.model.body_parentid[muj_id]
        # ret['j_st_id'][i] = sim.model.body_jntadr[muj_id]
        rel_pos = np.dot(muj_global_inertial_SO3[muj_pid], muj_global_inertial_pos[muj_id] - muj_global_inertial_pos[muj_pid])
        rel_SO3 = np.dot(muj_global_inertial_SO3[muj_pid].T, muj_global_inertial_SO3[muj_id])
        rel_SE3 = tf.squeeze(Rp_to_SE3(tf.expand_dims(tf.convert_to_tensor(rel_SO3,tf.float32),0), tf.expand_dims(tf.convert_to_tensor(rel_pos,tf.float32),0)), 0)
        ret['Mlist'][i] = rel_SE3

        G = np.diag(sim.model.body_inertia[muj_id])
        mI = sim.model.body_mass[muj_id] * np.eye(3)
        Gi = np.zeros((6,6))
        Gi[0:3,0:3] = G
        Gi[3:6,3:6] = mI
        ret['Glist'][i] = tf.convert_to_tensor(Gi, tf.float32)
    # ret['j_st_id'] = tf.stack(ret['j_st_id'],0)
    ret['Mlist'] = tf.stack(ret['Mlist'],0)
    ret['Glist'] = tf.stack(ret['Glist'],0)

    # ret['b_id'] = [None]*nq
    ret['Slist'] = [None]*nq
    for i in range(nq):
        # ret['b_id'][i] = muj_body_id.index(sim.model.jnt_bodyid[i])
        R = muj_global_joint_SO3[i]
        p = muj_global_joint_pos[i]
        Tsj = Rp_to_SE3(tf.expand_dims(tf.convert_to_tensor(R,tf.float32),0), tf.expand_dims(tf.convert_to_tensor(p,tf.float32),0))
        adTsj = tf.squeeze(adjoint(Tsj),0)
        if sim.model.jnt_type[i] == 2:
            screw_axis = np.concatenate([np.zeros(3), sim.model.jnt_axis[i]], axis=0)
        elif sim.model.jnt_type[i] == 3:
            screw_axis = np.concatenate([sim.model.jnt_axis[i], np.zeros(3)], axis=0)
        else:
            raise ValueError("Wrong Joint Type")
        screw_axis = tf.expand_dims(tf.convert_to_tensor(screw_axis, tf.float32),axis=1)
        S = tf.squeeze(tf.matmul(adTsj, screw_axis),1)
        ret['Slist'][i] = S
    # ret['b_id'] = tf.stack(ret['b_id'],0)
    ret['Slist'] = tf.stack(ret['Slist'],0)

    ret['init_ee_SE3'] = dict()
    ret['Bidlist'] = dict()
    ret['Blist'] = dict()
    for ee in ee_list:
        p = tf.expand_dims(tf.convert_to_tensor(muj_global_body_pos[sim.model.body_name2id(ee)], tf.float32),0)
        R = tf.expand_dims(tf.convert_to_tensor(muj_global_body_SO3[sim.model.body_name2id(ee)], tf.float32),0)
        ret['init_ee_SE3'][ee] = tf.squeeze(Rp_to_SE3(R,p),0)
        ret['Bidlist'][ee] = [None]*len(Bidlist[ee])
        ret['Blist'][ee] = [None]*len(Bidlist[ee])
        for i, id in enumerate(Bidlist[ee]):
            ret['Bidlist'][ee][i] = id
            R = muj_global_joint_SO3[id]
            p = muj_global_joint_pos[id]
            Tsj = tf.squeeze(Rp_to_SE3(tf.expand_dims(tf.convert_to_tensor(R,tf.float32),0), tf.expand_dims(tf.convert_to_tensor(p,tf.float32),0)),0)
            Tsb = ret['init_ee_SE3'][ee]
            Tbs = tf.squeeze(SE3_inv(tf.expand_dims(Tsb,0)))
            Tbj = tf.matmul(Tbs,Tsj)
            adTbj = tf.squeeze(adjoint(tf.expand_dims(Tbj,0)),0)
            if sim.model.jnt_type[i] == 2:
                screw_axis = np.concatenate([np.zeros(3), sim.model.jnt_axis[i]], axis=0)
            elif sim.model.jnt_type[i] == 3:
                screw_axis = np.concatenate([sim.model.jnt_axis[i], np.zeros(3)], axis=0)
            else:
                raise ValueError("Wrong Joint Type")
            screw_axis = tf.expand_dims(tf.convert_to_tensor(screw_axis, tf.float32),axis=1)
            B = tf.squeeze(tf.matmul(adTbj, screw_axis),1)
            ret['Blist'][ee][i] = B
        ret['Bidlist'][ee] = tf.stack(ret['Bidlist'][ee],0)
        ret['Blist'][ee] = tf.stack(ret['Blist'][ee],0)

    if verbose:
        print("="*80)
        print("Infos about Return Value")
        pretty_print_dictionary(ret)
        print("="*80)

    return ret

if __name__ == "__main__":
    # initial_config_from_mjcf(os.getcwd()+'/examples/assets/two_link_manipulator.xml', ['ee_b2'])
    # initial_config_from_mjcf(os.getcwd()+'/examples/assets/five_link_manipulator.xml', ['ee_b5', 'ee_b4'])
    initial_config_from_mjcf(os.getcwd()+'/examples/assets/my_hopper.xml', ['foot_sole'])
    # initial_config_from_mjcf(os.getcwd()+'/examples/assets/humanoid.xml', ['foot_sole'])
