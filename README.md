# tf_rbdl
tf_rbdl is a python library that contains some essential rigid body dynamics
algorithms such as forward dynamics and inverse dynamics using Tensorflow. It
uses Lie Group representation and provides kinematic and dynamics quatities of
rigid body systems.

## Examples
### General Usages
- APIs: [examples/general_usage.py](https://github.com/junhyeokahn/tf_rbdl/blob/master/examples/computation_time.py)
### Kinematics
- Function usages
- Inverse kinematics control
### Dynamics
- Function usages
- Inverse dynamics control

## Computation Efficiency
Running [examples/computation_time.py](https://github.com/junhyeokahn/tf_rbdl/blob/master/examples/computation_time.py) evaluates inverse dynamics for 5 batches and 500 epochs.

|     | Single Input w/ Autograph | Single Input w/o Autograph | Batch Input w/ Autograph | Batch Input w/o Autograph |
|:---:|:-------------------------:|:--------------------------:|:------------------------:|:-------------------------:|
| CPU |          3.82 (s)         |          71.68 (s)         |         0.95 (s)         |         13.02 (s)         |
| GPU |          8.53 (s)         |          89.75 (s)         |         2.46 (s)         |         18.27 (s)         |

## Installation
You can install `tf_rbd` from PyPI:
```bash
$ pip install tf_rbdl
```
or, you can also install from source:
```bash
$ git clone https://github.com/junhyeokahn/tf_rbdl
$ cd tf_rbdl
$ pip install -e .
```

## Todo
- [ ] Write examples
- [ ] Parse urdf

## Acknowledgement
This library is based on the book [Modern
Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) and the
[code](https://github.com/NxRLab/ModernRobotics).
