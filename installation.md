# Installation Instructions

:trollface:

## Installing the dependencies/packages
### Using Conda

To setup a conda environment with the required packages the following steps were used in a Linux terminal:

```
$ conda create -n environment_name python=3.7 anaconda
$ conda install -c conda-forge opencv=4.1.0
$ conda install -c conda-forge numpy=1.15.4
$ conda install -c conda-forge scipy=1.1.0
$ conda install -c conda-forge filterpy=1.4.1
$ conda install -c conda-forge numba=0.39.0
$ conda install -c conda-forge scikit-image=0.14.0
$ conda install -c conda-forge scikit-learn=0.19.2
$ conda install pytorch=1.0.1 torchvision=0.2.1 cudatoolkit=9.0 -c pytorch
$ git clone https://github.com/PurdueCAM2Project/HumanBehaviorS19.git
```

### Using Pip

To install all the required packages using pip, use the following commands:
``` 
$ git clone https://github.com/PurdueCAM2Project/HumanBehaviorS19.git
$ cd sort_tracker
$ pip3 install -r requirements.txt
$ pip3 install torch torchvision
```

---

**NOTE:**

This assumes that CUDA 9 has already been setup in the machine.

---
