# tf_rbdl
tf_rbdl is a python library that contains some essential rigid body dynamics
algorithms such as forward dynamics and inverse dynamics using Tensorflow. It
uses Lie Group representation and provides kinematic and dynamics quatities of
rigid body systems.

## Examples
### Kinematics
- Function usages
- Inverse kinematics control
### Dynamics
- Function usages
- Inverse dynamics control
### Misc
- Computation time

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

## Citation
If you use this code please cite it as:

```
@misc{tf_rbdl,
  title = {{tf_rbdl}: A Rigid Body Dyanmics Library using Tensorflow},
  author = "{Junhyeok Ahn}",
  howpublished = {\url{https://github.com/junhyeokahn/tf_rbdl}},
  url = "https://github.com/junhyeokahn/tf_rbdl",
  year = 2020,
  note = "[Online; accessed **-**-2020]"
}
```


## Acknowledgement
This code is based on the book [Modern
Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) and the
[code](https://github.com/NxRLab/ModernRobotics).
