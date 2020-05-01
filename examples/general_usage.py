import os
import sys
sys.path.append(os.getcwd())
import time

import numpy as np
import tensorflow as tf

import tf_rbdl.tf_rbdl as tf_rbd
import tf_rbdl.rbdl as rbd

# tf.config.experimental_run_functions_eagerly(True)
print("="*80)
print("Start")
print("="*80)
################################################################################
print("="*80)
print("1")
z = -1e-7
z_ = tf.constant([[0.0001], [1.], [2.]])
print("Batch")
print(tf_rbd.near_zero(z_))
assert(rbd.near_zero(z))
################################################################################
print("="*80)
print("2")
V1 = np.array([1.,2.,3.], dtype=np.float32)
V2 = tf.constant([1.,2.,3.], dtype=tf.float32)
V_ = tf.constant([[1.,2.,3.], [1.,2.,3.], [4., 5., 6.]], dtype=tf.float32)
assert(np.isclose(rbd.normalize_vector(V1), rbd.normalize_vector(V2).numpy()).all())
print(rbd.normalize_vector(V1))
print("Should be")
print([0.2672612419124244, 0.5345224838248488, 0.8017837257372732])
print("Batch")
print(tf_rbd.normalize_vector(V_))
################################################################################
print("="*80)
print("3")
R1 = np.array([[0., 0., 1.],[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
R2 = tf.constant([[0., 0., 1.],[1., 0., 0.], [0., 1., 0.]], dtype=tf.float32)
R3 = tf.constant([[1., 2., 3.],[4., 5., 6.], [7., 8., 9.]], dtype=tf.float32)
R_ = tf.stack([R1, R2, R3], axis=0)
assert(np.isclose(rbd.SO3_inv(R1), rbd.SO3_inv(R2).numpy()).all())
print(rbd.SO3_inv(R1))
print("Should be")
print(np.array([[0, 1, 0], [0, 0, 1],[1, 0, 0]]))
print("Batch")
batch_sol = tf_rbd.SO3_inv(R_)
print(batch_sol)
################################################################################
print("="*80)
print("4")
omg1 = np.array([1., 2., 3.], dtype=np.float32)
omg2 = tf.constant([1., 2., 3.], dtype=tf.float32)
omg3 = tf.constant([4., 5., 6.], dtype=tf.float32)
omg4 = tf.constant([7., 8., 9.], dtype=tf.float32)
omg_ = tf.stack([omg1, omg2, omg3, omg4], axis=0)
assert(np.isclose(rbd.vec_to_so3(omg1), rbd.vec_to_so3(omg2).numpy()).all())
print(rbd.vec_to_so3(omg1))
print("Should be")
print(np.array([[ 0, -3,  2],[ 3,  0, -1],[-2,  1,  0]]))
print("Batch")
print(tf_rbd.vec_to_so3(omg_))
################################################################################
print("="*80)
print("5")
so31 = np.array([[0., -3., 2.],[3., 0., -1.], [-2., 1., 0.]], dtype=np.float32)
so32 = tf.constant([[0., -3., 2.],[3., 0., -1.], [-2., 1., 0.]], dtype=tf.float32)
so33 = tf.constant([[0., -4., 3.],[4., 0., -2.], [-3., 2., 0.]], dtype=tf.float32)
so3_ = tf.stack([so31, so32, so33], axis=0)
assert(np.isclose(rbd.so3_to_vec(so31), rbd.so3_to_vec(so32).numpy()).all())
print(rbd.so3_to_vec(so31))
print("Should be")
print([1, 2, 3])
print("Batch")
print(tf_rbd.so3_to_vec(so3_))
################################################################################
print("="*80)
print("6")
expc31 = np.array([1., 2., 3.], dtype=np.float32)
expc32 = tf.constant([1., 2., 3.], dtype=tf.float32)
expc_ = tf.stack([expc31, expc32], axis=0)
assert(np.isclose(rbd.angvel_to_axis_ang(expc31)[0], rbd.angvel_to_axis_ang(expc32)[0].numpy()).all())
assert(np.isclose(rbd.angvel_to_axis_ang(expc31)[1], rbd.angvel_to_axis_ang(expc32)[1].numpy()).all())
print(rbd.angvel_to_axis_ang(expc31))
print("Should be")
print(([0.2672612419124244, 0.5345224838248488, 0.8017837257372732],3.7416573867739413) )
print("Batch")
print(tf_rbd.angvel_to_axis_ang(expc_))
################################################################################
print("="*80)
print("7")
so3mat1 = np.array([[ 0, -3,  2],[ 3,  0, -1],[-2,  1,  0]], dtype=np.float32)
so3mat2 = tf.constant([[ 0, -3,  2],[ 3,  0, -1],[-2,  1,  0]], dtype=tf.float32)
assert(np.isclose(rbd.so3_to_SO3(so3mat1), rbd.so3_to_SO3(so3mat2).numpy()).all())
print(rbd.so3_to_SO3(so3mat1))
print("Should be")
print(np.array([[-0.69492056,  0.71352099,  0.08929286],[-0.19200697, -0.30378504,  0.93319235],[ 0.69297817,  0.6313497 ,  0.34810748]]))
print("Batch")
so3mat_ = tf.stack([so3mat1, tf.zeros((3,3)), so3mat2, tf.zeros((3,3))], axis=0)
print("1")
print(tf_rbd.so3_to_SO3(so3mat_))
print(tf_rbd.so3_to_SO3_(so3mat_))
so3mat__ = tf.stack([so3mat1, tf.zeros((3,3)), so3mat2], axis=0)
print("2")
print(tf_rbd.so3_to_SO3(so3mat__))
print(tf_rbd.so3_to_SO3_(so3mat__))
so3mat___ = tf.stack([so3mat1, so3mat2], axis=0)
print("1")
print(tf_rbd.so3_to_SO3(so3mat___))
print(tf_rbd.so3_to_SO3_(so3mat___))
so3mat____ = tf.stack([tf.zeros((3,3)), tf.zeros((3,3))], axis=0)
print("1")
print(tf_rbd.so3_to_SO3(so3mat____))
print(tf_rbd.so3_to_SO3_(so3mat____))

################################################################################
print("="*80)
print("8")
R1 = np.array([[0, 0, 1],[1, 0, 0],[0, 1, 0]], np.float32)
R2 = tf.constant([[0, 0, 1],[1, 0, 0],[0, 1, 0]], tf.float32)
assert(np.isclose(rbd.SO3_to_so3(R1), rbd.SO3_to_so3(R2).numpy()).all())
print(rbd.SO3_to_so3(R1))
print("Should be")
print(np.array([[          0, -1.20919958,  1.20919958],[ 1.20919958,           0, -1.20919958],[-1.20919958,  1.20919958,           0]]))
print("Batch")
##
R1_ = tf.eye(3)
R21_ = tf.constant([[-1,0,0],[0,0,0],[0,0,0]], tf.float32)
R22_ = tf.constant([[0,0,0],[0,0,0],[0,0,-1]], tf.float32)
R23_ = tf.constant([[1,0,0],[0,-1,0],[0,0,-1]], tf.float32)
R3_ = tf.constant([[0, 0, 1],[1, 0, 0],[0, 1, 0]], tf.float32)
print((tf_rbd.SO3_to_so3(tf.stack([R1_, R1_], axis=0))))
print((tf_rbd.SO3_to_so3(tf.stack([R21_, R21_], axis=0))))
print((tf_rbd.SO3_to_so3(tf.stack([R22_, R22_], axis=0))))
print((tf_rbd.SO3_to_so3(tf.stack([R23_, R23_], axis=0))))
print((tf_rbd.SO3_to_so3(tf.stack([R3_, R3_], axis=0))))
print((tf_rbd.SO3_to_so3(tf.stack([R1_, R21_, R22_], axis=0))))
print((tf_rbd.SO3_to_so3(tf.stack([R1_, R21_, R23_], axis=0))))
print((tf_rbd.SO3_to_so3(tf.stack([R22_, R21_, R23_], axis=0))))
print((tf_rbd.SO3_to_so3(tf.stack([R1_, R22_, R3_], axis=0))))
print((tf_rbd.SO3_to_so3(tf.stack([R21_, R22_, R3_], axis=0))))
print((tf_rbd.SO3_to_so3(tf.stack([R1_, R1_, R3_], axis=0))))

################################################################################
print("="*80)
print("9")
R1 = np.array([[1, 0,  0], [0, 0, -1], [0, 1,  0]], np.float32)
p1 = np.array([1, 2, 5], np.float32)
R2 = tf.constant([[1, 0,  0], [0, 0, -1], [0, 1,  0]], tf.float32)
p2 = tf.constant([1, 2, 5], tf.float32)
assert(np.isclose(rbd.Rp_to_SE3(R1, p1), rbd.Rp_to_SE3(R2, p2).numpy()).all())
print(rbd.Rp_to_SE3(R1, p1))
print("Should be")
print(np.array([[1, 0,  0, 1],[0, 0, -1, 2],[0, 1,  0, 5],[0, 0,  0, 1]]))
print("Batch")
R_ = tf.stack([R2, R2], axis=0)
p_ = tf.stack([p2, p2], axis=0)
print(tf_rbd.Rp_to_SE3(R_, p_))
################################################################################
print("="*80)
print("10")
T1 = np.array([[1, 0,  0, 0],[0, 0, -1, 0],[0, 1,  0, 3],[0, 0,  0, 1]], np.float32)
T2 = tf.constant([[1, 0,  0, 0],[0, 0, -1, 0],[0, 1,  0, 3],[0, 0,  0, 1]], tf.float32)
assert(np.isclose(rbd.SE3_to_Rp(T1)[0], rbd.SE3_to_Rp(T2)[0].numpy()).all())
assert(np.isclose(rbd.SE3_to_Rp(T1)[1], rbd.SE3_to_Rp(T2)[1].numpy()).all())
print(rbd.SE3_to_Rp(T1))
print("Should be")
print(([[1, 0,  0], [0, 0, -1], [0, 1,  0]],  [0, 0, 3]))
print("Batch")
T_ = tf.stack([T2,T2],axis=0)
print(tf_rbd.SE3_to_Rp(T_))
################################################################################
print("="*80)
print("11")
T1 = np.array([[1, 0,  0, 0],[0, 0, -1, 0],[0, 1,  0, 3],[0, 0,  0, 1]], np.float32)
T2 = tf.constant([[1, 0,  0, 0],[0, 0, -1, 0],[0, 1,  0, 3],[0, 0,  0, 1]], tf.float32)
assert(np.isclose(rbd.SE3_inv(T1), rbd.SE3_inv(T2).numpy()).all())
print(rbd.SE3_inv(T1))
print("Should be")
print(np.array([[1,  0, 0,  0],[0,  0, 1, -3],[0, -1, 0,  0],[0,  0, 0,  1]]))
print("Batch")
T_ = tf.stack([T2, T2], axis=0)
print(tf_rbd.SE3_inv(T_))
################################################################################
print("="*80)
print("12")
V1 = np.array([1, 2, 3, 4, 5, 6], np.float32)
V2 = tf.constant([1, 2, 3, 4, 5, 6], tf.float32)
assert(np.isclose(rbd.vec_to_se3(V1), rbd.vec_to_se3(V2).numpy()).all())
print(rbd.vec_to_se3(V1))
print("Should be")
print(np.array([[ 0, -3,  2, 4], [ 3,  0, -1, 5], [-2,  1,  0, 6], [ 0,  0,  0, 0]]))
print("Batch")
V_ = tf.stack([V2, V2], axis=0)
print(tf_rbd.vec_to_se3(V_))
################################################################################
print("="*80)
print("13")
V1 = np.array([[ 0, -3,  2, 4], [ 3,  0, -1, 5], [-2,  1,  0, 6], [ 0,  0,  0, 0]], np.float32)
V2 = tf.constant([[ 0, -3,  2, 4], [ 3,  0, -1, 5], [-2,  1,  0, 6], [ 0,  0,  0, 0]], tf.float32)
assert(np.isclose(rbd.se3_to_vec(V1), rbd.se3_to_vec(V2).numpy()).all())
print(rbd.se3_to_vec(V1))
print("Should be")
print([1, 2, 3, 4, 5, 6])
print("Batch")
V_ = tf.stack([V2,V2],axis=0)
print(tf_rbd.se3_to_vec(V_))
################################################################################
print("="*80)
print("14")
V1 = np.array([[1, 0,  0, 0], [0, 0, -1, 0], [0, 1,  0, 3], [0, 0,  0, 1]], np.float32)
V2 = tf.constant([[1, 0,  0, 0], [0, 0, -1, 0], [0, 1,  0, 3], [0, 0,  0, 1]], tf.float32)
assert(np.isclose(rbd.adjoint(V1), rbd.adjoint(V2).numpy()).all())
print(rbd.adjoint(V1))
print("Should be")
print(np.array([[1, 0,  0, 0, 0,  0],[0, 0, -1, 0, 0,  0],[0, 1,  0, 0, 0,  0],[0, 0,  3, 1, 0,  0],[3, 0,  0, 0, 0, -1],[0, 0,  0, 0, 1,  0]]))
print("Batch")
V_ = tf.stack([V2,V2],axis=0)
print(tf_rbd.adjoint(V_))
################################################################################
print("="*80)
print("16")
V1 = np.array([[0,                 0,                  0,                 0],
          [0,                 0, -1.570796326794897, 2.356194490192345],
          [0, 1.570796326794897,                  0, 2.356194490192345],
          [0,                 0,                  0,                 0]], np.float32)
V2 = tf.constant([[0,                 0,                  0,                 0],
          [0,                 0, -1.570796326794897, 2.356194490192345],
          [0, 1.570796326794897,                  0, 2.356194490192345],
          [0,                 0,                  0,                 0]], tf.float32)
assert(np.isclose(rbd.se3_to_SE3(V1), rbd.se3_to_SE3(V2).numpy()).all())
print(rbd.se3_to_SE3(V1))
print("Should be")
print(np.array([[1.0, 0.0,  0.0, 0.0],[0.0, 0.0, -1.0, 0.0],[0.0, 1.0,  0.0, 3.0],[  0,   0,    0,   1]]))
print("Batch")
V3 = tf.zeros((4,4))
print(rbd.se3_to_SE3(V3))
V_ = tf.stack([V3,V3],axis=0)
print(tf_rbd.se3_to_SE3(V_))
print(tf_rbd.se3_to_SE3(V_))
print(tf_rbd.se3_to_SE3(V_))
################################################################################
print("="*80)
print("17")
V1 = np.array([[1,0,0,0], [0,0,-1,0], [0,1,0,3], [0,0,0,1]], np.float32)
V2 = tf.constant([[1,0,0,0], [0,0,-1,0], [0,1,0,3], [0,0,0,1]], tf.float32)
assert(np.isclose(rbd.SE3_to_se3(V1), rbd.SE3_to_se3(V2).numpy()).all())
print(rbd.SE3_to_se3(V1))
print("Should be")
print(np.array([[0, 0, 0, 0],[0,0,-1.5708,2.3562],[0,1.5708,0,2.3562],[0,0,0,0]]))
print("Batch")
V3 = tf.eye(4)
V_ = tf.stack([V3,V3],0)
print(rbd.SE3_to_se3(V3))
print(tf_rbd.SE3_to_se3(V_))
V_ = tf.stack([V2,V2],0)
V_ = tf.stack([V2,V3,V2,V3],0)
print(tf_rbd.SE3_to_se3(V_))
################################################################################
print("="*80)
print("18")
M1 = np.array([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]], np.float32)
M2 = tf.constant([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]], tf.float32)
Blist1 = [np.array([0, 0, -1, 2, 0, 0], np.float32),
          np.array([0, 0,  0, 0, 1, 0], np.float32),
          np.array([0, 0,  1, 0, 0, 0.1], np.float32)]
Blist2 = [tf.constant([0, 0, -1, 2, 0, 0], tf.float32),
          tf.constant([0, 0,  0, 0, 1, 0], tf.float32),
          tf.constant([0, 0,  1, 0, 0, 0.1], tf.float32)]
thetalist1 = np.array([np.pi / 2.0, 3, np.pi], np.float32)
thetalist2 = tf.constant([np.pi / 2.0, 3, np.pi], tf.float32)
assert(np.isclose(rbd.fk_body(M1, Blist1, thetalist1), rbd.fk_body(M2, Blist2, thetalist2).numpy(), 1e-4, 1e-7).all())
print(rbd.fk_body(M1, Blist1, thetalist1))
print("Should be")
print(np.array([[ -1.14423775e-17,   1.00000000e+00,   0.00000000e+00,  -5.00000000e+00],[  1.00000000e+00,   1.14423775e-17,   0.00000000e+00,   4.00000000e+00],[              0.,               0.,              -1.   ,    1.68584073],[              0.,               0.,               0.,               1.]]))
print("Batch")
th_ = tf.stack([thetalist2, thetalist2],0)
print(tf_rbd.fk_body(M2, Blist2, th_))

################################################################################
print("="*80)
print("19")
M1 = np.array([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]], np.float32)
M2 = tf.constant([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]], tf.float32)
Slist1 = [np.array([0, 0,  1, 4, 0, 0], np.float32),
          np.array([0, 0,  0, 0, 1, 0], np.float32),
          np.array([0, 0,  -1, -6, 0, -0.1], np.float32)]
Slist2 = [tf.constant([0, 0, 1, 4, 0, 0], tf.float32),
          tf.constant([0, 0,  0, 0, 1, 0], tf.float32),
          tf.constant([0, 0, -1, -6, 0, -0.1], tf.float32)]
thetalist1 = np.array([np.pi / 2.0, 3, np.pi], np.float32)
thetalist2 = tf.constant([np.pi / 2.0, 3, np.pi], tf.float32)
assert(np.isclose(rbd.fk_space(M1, Slist1, thetalist1), rbd.fk_space(M2, Slist2, thetalist2).numpy(), 1e-4, 1e-7).all())
print(rbd.fk_space(M1, Slist1, thetalist1))
print("Should be")
print(np.array([[ -1.14423775e-17 ,  1.00000000e+00  , 0.00000000e+00,  -5.00000000e+00],[  1.00000000e+00,   1.14423775e-17,   0.00000000e+00,   4.00000000e+00],[              0.,               0.,              -1.,       1.68584073],[              0.,               0.,               0.,               1.]]))
print("Batch")
th_ = tf.stack([thetalist2, thetalist2],0)
print(tf_rbd.fk_space(M2, Slist2, th_))
################################################################################
print("="*80)
print("20")
Blist1 = [np.array([0, 0, 1,   0, 0.2, 0.2], np.float32),
          np.array([1, 0, 0,   2,   0,   3], np.float32),
          np.array([0, 1, 0,   0,   2,   1], np.float32),
          np.array([1, 0, 0, 0.2, 0.3, 0.4], np.float32)]
Blist2 = [tf.constant([0, 0, 1,   0, 0.2, 0.2], tf.float32),
          tf.constant([1, 0, 0,   2,   0,   3], tf.float32),
          tf.constant([0, 1, 0,   0,   2,   1], tf.float32),
          tf.constant([1, 0, 0, 0.2, 0.3, 0.4], tf.float32)]
thetalist1 = np.array([0.2, 1.1, 0.1, 1.2], np.float32)
thetalist2 = tf.constant([0.2, 1.1, 0.1, 1.2], tf.float32)
assert(np.isclose(rbd.jac_body(Blist1, thetalist1), rbd.jac_body(Blist2, thetalist2).numpy()).all())
print(rbd.jac_body(Blist1, thetalist1))
print("Should be")
print(np.array([[-0.04528405 , 0.99500417 , 0.,          1.,        ],[ 0.74359313,  0.09304865,  0.36235775,  0.,        ],[-0.66709716,  0.03617541, -0.93203909,  0.        ],[ 2.32586047,  1.66809,     0.56410831,  0.2,       ],[-1.44321167 , 2.94561275,  1.43306521,  0.3       ],[-2.06639565,  1.82881722, -1.58868628,  0.4       ]]))
print("Batch")
th_ = tf.stack([thetalist2, thetalist2],0)
print(tf_rbd.jac_body(Blist2, th_))
################################################################################
print("="*80)
print("21")
Slist1 = [np.array([0, 0, 1,   0, 0.2, 0.2], np.float32),
          np.array([1, 0, 0,   2,   0,   3], np.float32),
          np.array([0, 1, 0,   0,   2,   1], np.float32),
          np.array([1, 0, 0, 0.2, 0.3, 0.4], np.float32)]
Slist2 = [tf.constant([0, 0, 1,   0, 0.2, 0.2], tf.float32),
          tf.constant([1, 0, 0,   2,   0,   3], tf.float32),
          tf.constant([0, 1, 0,   0,   2,   1], tf.float32),
          tf.constant([1, 0, 0, 0.2, 0.3, 0.4], tf.float32)]
thetalist1 = np.array([0.2, 1.1, 0.1, 1.2], np.float32)
thetalist2 = tf.constant([0.2, 1.1, 0.1, 1.2], tf.float32)
assert(np.isclose(rbd.jac_space(Slist1, thetalist1), rbd.jac_space(Slist2, thetalist2).numpy()).all())
print(rbd.jac_space(Slist1, thetalist1))
print("Should be")
print(np.array([[ 0. ,         0.98006658 ,-0.09011564,  0.95749426], [ 0.,          0.19866933  ,0.4445544,   0.28487557], [ 1.,          0.         , 0.89120736, -0.04528405], [ 0.,          1.95218638 ,-2.21635216, -0.51161537], [ 0.2,         0.43654132 ,-2.43712573,  2.77535713], [ 0.2,         2.96026613 , 3.23573065  ,2.22512443]]))
print("Batch")
th_ = tf.stack([thetalist2, thetalist2],0)
print(tf_rbd.jac_space(Blist2, th_))
################################################################################
print("="*80)
print("22")
Blist1 = [np.array([0, 0, -1,  2,   0.,  0.], np.float32),
          np.array([0, 0, 0,   0,   1,   0.], np.float32),
          np.array([0, 0, 1,   0,   0,   0.1], np.float32)]
Blist2 = [tf.constant([0, 0, -1,  2,   0.,  0.], tf.float32),
          tf.constant([0, 0, 0,   0,   1,   0.], tf.float32),
          tf.constant([0, 0, 1,   0,   0,   0.1],tf.float32)]
M1 = np.array([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]], np.float32)
M2 = tf.constant([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]], tf.float32)
T1 = np.array([[0, 1, 0, -5], [1, 0, 0, 4], [0, 0, -1, 1.6858], [0, 0, 0, 1]], np.float32)
T2 = tf.constant([[0, 1, 0, -5], [1, 0, 0, 4], [0, 0, -1, 1.6858], [0, 0, 0, 1]], tf.float32)
thetalist1 = np.array([1.5, 2.5, 3.], np.float32)
thetalist2 = tf.constant([1.5, 2.5, 3.], tf.float32)
eomg = 0.01
ev = 0.01
assert(np.isclose(rbd.ik_body(Blist1, M1, T1, thetalist1, eomg, ev)[0], rbd.ik_body(Blist2, M2, T2, thetalist2, eomg, ev)[0].numpy(), 1e-04, 1e-07).all())
print(rbd.ik_body(Blist1, M1, T1, thetalist1, eomg, ev))
print("Should be")
print([1.57073819, 2.999667, 3.14153913])
print("Batch")
T_ = tf.stack([T2,T2],axis=0)
th0_ = tf.stack([thetalist2,thetalist2],axis=0)
print(tf_rbd.ik_body(Blist2, M2, T_, th0_, eomg, ev))
################################################################################
print("="*80)
print("23")
Slist1 = [np.array([0, 0, 1,  4,   0.,  0.], np.float32),
          np.array([0, 0, 0,   0,   1,   0.], np.float32),
          np.array([0, 0, -1,   -6,   0,   -0.1], np.float32)]
Slist2 = [tf.constant([0, 0, 1,  4,   0.,  0.], tf.float32),
          tf.constant([0, 0, 0,   0,   1,   0.], tf.float32),
          tf.constant([0, 0, -1,   -6,   0,   -0.1], tf.float32)]

M1 = np.array([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]], np.float32)
M2 = tf.constant([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]], tf.float32)
T1 = np.array([[0, 1, 0, -5], [1, 0, 0, 4], [0, 0, -1, 1.6858], [0, 0, 0, 1]], np.float32)
T2 = tf.constant([[0, 1, 0, -5], [1, 0, 0, 4], [0, 0, -1, 1.6858], [0, 0, 0, 1]], tf.float32)
thetalist1 = np.array([1.5, 2.5, 3.], np.float32)
thetalist2 = tf.constant([1.5, 2.5, 3.], tf.float32)
eomg = 0.01
ev = 0.01
assert(np.isclose(rbd.ik_space(Slist1, M1, T1, thetalist1, eomg, ev)[0], rbd.ik_space(Slist2, M2, T2, thetalist2, eomg, ev)[0].numpy()).all())
print(rbd.ik_space(Slist1, M1, T1, thetalist1, eomg, ev))
print("Should be")
print([1.57073785, 2.99966405, 3.14154125])
print("Batch")
T_ = tf.stack([T2,T2],0)
th0_ = tf.stack([thetalist2, thetalist2], 0)
print(tf_rbd.ik_space(Slist2, M2, T_, th0_, eomg, ev))
################################################################################
print("="*80)
print("24")
V1 = np.array([1,2,3,4,5,6], np.float32)
V2 = tf.constant([1,2,3,4,5,6], tf.float32)
assert(np.isclose(rbd.ad(V1), rbd.ad(V2).numpy(), 1e-04, 1e-7).all())
print(rbd.ad(V1))
print("Should be")
print(np.array([[0, -3, 2, 0, 0, 0],
 [3, 0, -1, 0, 0, 0],
 [-2, 1, 0, 0, 0, 0],
 [0, -6, 5, 0, -3, 2],
 [6, 0, -4, 3, 0, -1],
 [-5, 4, 0, -2, 1, 0]]))
print("Batch")
V_ = tf.stack([V2,V2],0)
print(tf_rbd.ad(V_))
################################################################################
print("="*80)
print("25")
thetalist = np.array([0.1, 0.1, 0.1], np.float32)
dthetalist = np.array([0.1, 0.2, 0.3], np.float32)
ddthetalist = np.array([2, 1.5, 1], np.float32)
thetalist_ = tf.constant([0.1, 0.1, 0.1], tf.float32)
dthetalist_ = tf.constant([0.1, 0.2, 0.3], tf.float32)
ddthetalist_ = tf.constant([2, 1.5, 1], tf.float32)
g = np.array([0, 0, -9.8], np.float32)
g_ = tf.constant([0, 0, -9.8], tf.float32)
Ftip = np.array([1, 1, 1, 1, 1, 1], np.float32)
Ftip_ = tf.constant([1, 1, 1, 1, 1, 1], tf.float32)
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
assert(np.isclose(rbd.id_space(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist,Slist), rbd.id_space(thetalist_, dthetalist_, ddthetalist_, g_, Ftip_, Mlist_, Glist_, Slist_).numpy()).all())
print(rbd.id_space(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist))
print("Should be")
print([74.696161552874514, -33.067660158514578, -3.2305731379014242])
print("Batch")
th_ = tf.stack([thetalist_]*3,0)
dth_ = tf.stack([dthetalist_]*3,0)
ddth_ = tf.stack([ddthetalist_]*3,0)
Ft_ = tf.stack([Ftip_]*3,0)
print(tf_rbd.id_space(th_,dth_,ddth_,g_,Ft_,Mlist_,Glist_,Slist_))

################################################################################
print("="*80)
print("26")
thetalist = np.array([0.1, 0.1, 0.1], np.float32)
thetalist_ = tf.constant([0.1, 0.1, 0.1], tf.float32)
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
assert(np.isclose(rbd.mass_matrix(thetalist, Mlist, Glist, Slist), rbd.mass_matrix(thetalist_, Mlist_, Glist_, Slist_).numpy(), 1e-04, 1e-07).all())
print(rbd.mass_matrix(thetalist, Mlist, Glist, Slist))
print("Should be")
print(np.array([[  2.25433380e+01 , -3.07146754e-01,  -7.18426391e-03],[ -3.07146754e-01   ,1.96850717e+00,   4.32157368e-01],[ -7.18426391e-03   ,4.32157368e-01,   1.91630858e-01]]))
print("Batch")
print(tf_rbd.mass_matrix(th_, Mlist_, Glist_, Slist_))
################################################################################
print("="*80)
print("27")
thetalist = np.array([0.1, 0.1, 0.1], np.float32)
dthetalist = np.array([0.1, 0.2, 0.3], np.float32)
thetalist_ = tf.constant([0.1, 0.1, 0.1], tf.float32)
dthetalist_ = tf.constant([0.1, 0.2, 0.3], tf.float32)
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
assert(np.isclose(rbd.coriolis_forces(thetalist, dthetalist, Mlist, Glist, Slist), rbd.coriolis_forces(thetalist_, dthetalist_, Mlist_, Glist_, Slist_).numpy()).all())
print(rbd.coriolis_forces(thetalist, dthetalist, Mlist, Glist, Slist))
print("Should be")
print([0.26453118054501235, -0.055051568289165499, -0.0068913200682489129])
print("Batch")
print(tf_rbd.coriolis_forces(th_, dth_,Mlist_,Glist_, Slist_))
################################################################################
print("="*80)
print("28")
thetalist = np.array([0.1, 0.1, 0.1], np.float32)
thetalist_ = tf.constant([0.1, 0.1, 0.1], tf.float32)
Ftip = np.array([1,1,1,1,1,1], np.float32)
Ftip_ = tf.convert_to_tensor(Ftip, tf.float32)
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
assert(np.isclose(rbd.end_effector_forces(thetalist, Ftip, Mlist, Glist, Slist), rbd.end_effector_forces(thetalist_, Ftip_, Mlist_, Glist_, Slist_).numpy()).all())
print(rbd.end_effector_forces(thetalist, Ftip, Mlist, Glist, Slist))
print("Should be")
print([1.4095460782639782, 1.8577149723180628, 1.392409])
################################################################################
print("="*80)
print("29")
thetalist = np.array([0.1, 0.1, 0.1], np.float32)
dthetalist = np.array([0.1, 0.2, 0.3], np.float32)
thetalist_ = tf.constant([0.1, 0.1, 0.1], tf.float32)
dthetalist_ = tf.constant([0.1, 0.2, 0.3], tf.float32)
taulist = np.array([0.5, 0.6, 0.7], np.float32)
taulist_ = tf.constant([0.5, 0.6, 0.7], tf.float32)
g = np.array([0., 0., -9.8], np.float32)
g_ = tf.constant([0., 0., -9.8], tf.float32)
Ftip = np.array([1,1,1,1,1,1], np.float32)
Ftip_ = tf.convert_to_tensor(Ftip, tf.float32)
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
assert(np.isclose(rbd.fd_space(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist), rbd.fd_space(thetalist_, dthetalist_, taulist_, g_, Ftip_, Mlist_, Glist_, Slist_).numpy()).all())
print(rbd.fd_space(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist))
print("Should be")
print([ -0.97392907 , 25.58466784, -32.91499212])
print("Batch")
th_ = tf.stack([thetalist_]*3,0)
dth_ = tf.stack([dthetalist_]*3,0)
tau_ = tf.stack([taulist_]*3,0)
Ft_ = tf.stack([Ftip_]*3,0)
print(tf_rbd.fd_space(th_,dth_,tau_,g_,Ft_,Mlist_,Glist_,Slist_))
################################################################################
print("="*80)
print("30")
thetalist = np.array([0.1,0.1,0.1], np.float32)
dthetalist = np.array([0.1,0.2,0.3], np.float32)
ddthetalist = np.array([2,1.5,1], np.float32)
dt = 0.1
thetalist_ = tf.constant([0.1,0.1,0.1], tf.float32)
dthetalist_ = tf.constant([0.1,0.2,0.3], tf.float32)
ddthetalist_ = tf.constant([2,1.5,1], tf.float32)
aa1, bb1 = rbd.euler_step(thetalist, dthetalist, ddthetalist, dt)
aa2, bb2 = rbd.euler_step(thetalist_, dthetalist_, ddthetalist_, dt)
assert(np.isclose(aa1, aa2.numpy()).all())
assert(np.isclose(bb1, bb2.numpy()).all())
print(rbd.euler_step(thetalist, dthetalist, ddthetalist, dt))
print("Should be")
print([ 0.11,  0.12,  0.13], [ 0.3 ,  0.35,  0.4 ])
print("Batch")

th_ = tf.stack([thetalist_]*3,0)
dth_ = tf.stack([dthetalist_]*3,0)
ddth_ = tf.stack([ddthetalist_]*3,0)
print(tf_rbd.euler_step(th_, dth_, ddth_, dt))

################################################################################
tf.config.experimental_run_functions_eagerly(False)
print("Done")
