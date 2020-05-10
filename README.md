[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3819602.svg)](https://doi.org/10.5281/zenodo.3819602)

# tf_rbdl
tf_rbdl is a python library that contains some essential rigid body dynamics
algorithms such as forward dynamics and inverse dynamics using Tensorflow. It
uses Lie Group representation and provides kinematic and dynamics quatities of
rigid body systems.

## Examples
### General Usages
- See [examples/general_usage.py](https://github.com/junhyeokahn/tf_rbdl/blob/master/examples/general_usage.py) for details.

## Installation
You can install `tf_rbdl` from PyPI:
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
```
@misc{junhyeok_ahn_2020_3780707,
  author       = {Junhyeok Ahn},
  title        = {{tf\_rbdl: Rigid Body Dynamics Library using 
                   Tensorflow}},
  month        = may,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.3780707},
  url          = {https://doi.org/10.5281/zenodo.3780707}
}
```

## Todo
- [ ] Write examples
- [ ] Parse urdf

## Acknowledgement
This library is based on the book [Modern
Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) and the
[code](https://github.com/NxRLab/ModernRobotics).
