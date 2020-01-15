# Utility Function

In this folder there are tree script which add additional and separate functionality to the application. 

## Analysis

This script give to the user the opportunity to build a reward hit-map manually. Using terminal, the user uses the keyboard commands to move the agent within the selected environment. The commands to use are:

- ↑ : move forward

- → : turn rigth

- ← : turn left

## Testing

This script is usefull to understand if the policy model is trained correctly. The user has to specify the environment and the policy weigth path before launching the script with terminal.

## Training
 
This usefull script is used to understand if the reward model structure or the choosen hyperparameters are good or not. This script represent the entire algorithm described in the [paper](http://papers.nips.cc/paper/8025-reward-learning-from-human-preferences-and-demonstrations-in-atari) using a [oralce](../ReinforcementLearning/ReinforcemenLearning.md) to make the preferencies. 
For now this script can be used to test models in the `MiniGrid-Empty-6x6-v0` environment.


