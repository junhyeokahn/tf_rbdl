from tqdm import tqdm
import numpy as np
from mujoco_py import load_model_from_path, MjSim, functions, MjViewer
import numpy as np
np.set_printoptions(precision=4)
import os
import sys
sys.path.append(os.getcwd())

from tf_rbdl.utils import *
from tf_rbdl.kinematics import *
from tf_rbdl.dynamics import *
from tf_rbdl.liegroup import *

def general_usage(xml_path,ee_list,N):
    # ==========================================================================
    # Quantaties
    # ==========================================================================
    ee_SE3_mujoco, ee_jac_mujoco, M_mujoco, qfrc_bias_mujoco = dict(), dict(), [], []
    ee_SE3_tf_rbdl, ee_jac_tf_rbdl, M_tf_rbdl, qfrc_bias_tf_rbdl= dict(), dict(), None, None
    for ee in ee_list:
        ee_SE3_mujoco[ee], ee_jac_mujoco[ee], ee_SE3_tf_rbdl[ee], ee_jac_tf_rbdl[ee] = [],[],None,None

    # ==========================================================================
    # Mujoco
    # ==========================================================================
    mujoco_model = load_model_from_path(xml_path)
    m = MjSim(mujoco_model)
    m.forward()
    sim_state=m.get_state()
    q = np.random.uniform(-1., 1., (N,m.model.nq))
    qdot = np.random.uniform(-1., 1., (N,m.model.nq))

    for i in tqdm(range(N)):
        for j in range(m.model.nq):
            sim_state.qpos[j] = q[i,j]
            sim_state.qvel[j] = qdot[i,j]
        m.set_state(sim_state)
        m.forward()
        for ee in ee_list:
            T_ee = np.eye(4)
            T_ee[0:3, 0:3] = m.data.get_body_xmat(ee)
            T_ee[0:3, 3] = m.data.get_body_xpos(ee)
            ee_SE3_mujoco[ee].append(T_ee)

            J_ee = np.zeros((6, m.model.nq))
            jacp = np.zeros(m.model.nv*3)
            jacr = np.zeros(m.model.nv*3)
            functions.mj_jac(m.model, m.data, jacp, jacr, m.data.get_body_xpos(ee), m.model.body_name2id(ee))
            J_ee[0:3,:] = jacr.reshape(3, m.model.nv)
            J_ee[3:6,:] = jacp.reshape(3, m.model.nv)
            ee_jac_mujoco[ee].append(J_ee)
        M_ = np.zeros(m.model.nv * m.model.nv)
        functions.mj_fullM(m.model, M_, m.data.qM)
        M_mujoco.append(M_.reshape(m.model.nq,m.model.nq))
        qfrc_bias_mujoco.append(np.copy(m.data.qfrc_bias))

    # ==========================================================================
    # tf_rbdl
    # ==========================================================================
    ic = initial_config_from_mjcf(xml_path, ee_list, verbose=True)
    for ee in ee_list:
        ee_SE3_tf_rbdl[ee] = fk(ic['init_ee_SE3'][ee], ic['Blist'][ee], ic['Bidlist'][ee],tf.convert_to_tensor(q,tf.float32))
        ee_local_jac_tf_rbdl = jacobian(ic['Blist'][ee], ic['Bidlist'][ee],tf.convert_to_tensor(q,tf.float32))
        R = ee_SE3_tf_rbdl[ee][:,0:3,0:3]
        RR = tf.concat([tf.concat([R, tf.zeros((N,3,3))],2), tf.concat([tf.zeros((N,3,3)), R],2)],1)
        ee_jac_tf_rbdl[ee] = tf.matmul(RR, ee_local_jac_tf_rbdl)
    M_tf_rbdl = mass_matrix(tf.convert_to_tensor(q,tf.float32),ic['pidlist'], ic['Mlist'], ic['Glist'], ic['Slist']) + tf.eye(m.model.nq)*ic['joint_armature']

    cori = coriolis_forces(tf.convert_to_tensor(q,tf.float32), tf.convert_to_tensor(qdot,tf.float32),ic['pidlist'], ic['Mlist'], ic['Glist'], ic['Slist'])
    grav = gravity_forces(tf.convert_to_tensor(q,tf.float32), ic['g'],ic['pidlist'], ic['Mlist'], ic['Glist'], ic['Slist'])
    qfrc_bias_tf_rbdl = cori + grav

    # ==========================================================================
    # Comparing
    # ==========================================================================

    # print("="*80)
    # print("q")
    # print(q)
    # print("qdot")
    # print(qdot)
    # print("="*80)
    # for i in range(N):
        # for ee in ee_list:
            # print("-"*80)
            # print("{} SE3".format(ee))
            # print("Mujoco\n{}\ntf_rbdl\n{}\nComparison\n{}".format(ee_SE3_mujoco[ee][i], ee_SE3_tf_rbdl[ee][i].numpy(), np.isclose(ee_SE3_mujoco[ee][i], ee_SE3_tf_rbdl[ee][i].numpy(),atol=1e-5)))
            # print("-"*80)
            # print("{} Jacobian".format(ee))
            # print("Mujoco\n{}\ntf_rbdl\n{}\nComparison\n{}".format(ee_jac_mujoco[ee][i], ee_jac_tf_rbdl[ee][i].numpy(), np.isclose(ee_jac_mujoco[ee][i], ee_jac_tf_rbdl[ee][i].numpy(),atol=1e-5)))
        # print("-"*80)
        # print("Mass Matrix")
        # print("Mujoco\n{}\ntf_rbdl\n{}\nComparison\n{}".format(M_mujoco[i], M_tf_rbdl[i].numpy(), np.isclose(M_mujoco[i], M_tf_rbdl[i].numpy(),atol=1e-5)))
        # print("-"*80)
        # print("Bias frc")
        # print("Mujoco\n{}\ntf_rbdl\n{}\nComparison\n{}".format(qfrc_bias_mujoco[i], qfrc_bias_tf_rbdl[i].numpy(), np.isclose(qfrc_bias_mujoco[i], qfrc_bias_tf_rbdl[i].numpy(),atol=1e-5)))
    # print("="*80)

    for i in range(N):
        for ee in ee_list:
            assert np.allclose(ee_SE3_mujoco[ee][i], ee_SE3_tf_rbdl[ee][i].numpy(),atol=1e-04), "Mujoco\n{}\ntf_rbdl\n{}\nComparison\n{}".format(ee_SE3_mujoco[ee][i], ee_SE3_tf_rbdl[ee][i].numpy(), np.isclose(ee_SE3_mujoco[ee][i], ee_SE3_tf_rbdl[ee][i].numpy()))
            assert np.allclose(ee_jac_mujoco[ee][i], ee_jac_tf_rbdl[ee][i].numpy(),atol=1e-04), "Mujoco\n{}\ntf_rbdl\n{}\nComparison\n{}".format(ee_jac_mujoco[ee][i], ee_jac_tf_rbdl[ee][i].numpy(), np.isclose(ee_jac_mujoco[ee][i], ee_jac_tf_rbdl[ee][i].numpy()))
        assert np.allclose(M_mujoco[i], M_tf_rbdl[i].numpy(),atol=1e-04), "Mujoco\n{}\ntf_rbdl\n{}\nComparison\n{}".format(M_mujoco[i], M_tf_rbdl[i].numpy(), np.isclose(M_mujoco[i], M_tf_rbdl[i].numpy()))
        assert np.allclose(qfrc_bias_mujoco[i], qfrc_bias_tf_rbdl[i].numpy(),atol=1e-04), "Mujoco\n{}\ntf_rbdl\n{}\nComparison\n{}".format(qfrc_bias_mujoco[i], qfrc_bias_tf_rbdl[i].numpy(), np.isclose(qfrc_bias_mujoco[i], qfrc_bias_tf_rbdl[i].numpy()))

if __name__ == "__main__":
    # tf.config.experimental_run_functions_eagerly(True)
    general_usage(os.getcwd()+'/examples/assets/my_reacher.xml', ['finger'], 2000)
    # general_usage(os.getcwd()+'/examples/assets/two_link_manipulator.xml', ['ee_b2'], 1)
    # general_usage(os.getcwd()+'/examples/assets/five_link_manipulator.xml', ['ee_b4', 'ee_b5'], 2000)
    # general_usage(os.getcwd()+'/examples/assets/my_hopper.xml', ['foot_sole'], 2000)
    # tf.config.experimental_run_functions_eagerly(False)
