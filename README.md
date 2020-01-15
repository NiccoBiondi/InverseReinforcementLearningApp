# Inverse Reinforcement Learning Application

The aim of this project is to create a simple application that can give the opportunity to all the user to use the Inverse Reinforcement Learning technique. The application is implemented in Python and the interface is made entirely with PyQt5.


<p align="center">
  <img src="Build/gitimages/main_view.png" width="400"/>
  <img src="Build/gitimages/alg_view.png" width="400"/>
</p>

## Feature

Features include:

- The opportunity to choose from [different environment](Widget/SETUP.md).
- Auto save function during the work.
- [History window](Widget/SETUP.md) to memorize the user preferencies.
- Restore previous work sesssion before starting a new work.
- [Oracle](ReinforcementLearning/ReinforcemenLearning.md) to speed up the annotations and to debug the policy and reward model.
- [Utility function](Utility/UTILS.md) which the user can use to analyze the reward model and to testing the policy and reward model too. 

## Dependecies

- Pytorch
- NumPy
- OpenAI Gym
- Matplotlib 
- opencv
- PIL

For detailed steps to install PyTorch, follow the [PyTorch installation instruction](https://pytorch.org/get-started/locally/). A typical user can install PyTorch using the following commands:

```bash
# Create virtual environment
conda create -n minigrid python=3.7

# Activate virtual environment
conda activate minigrid

# Install OpenAI Gym
pip3 install gym-minigrid

# Install PyTorch
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

For detailed steps to install OpenAI Gym follow the Installation steps in [gym-minigrid github repository](https://github.com/maximecb/gym-minigrid).

The other packet can be installed using pip:


```bash
conda activate minigrid
pip install numpy
pip install mathplotlib
pip install Pillow
pip install opencv-python
```