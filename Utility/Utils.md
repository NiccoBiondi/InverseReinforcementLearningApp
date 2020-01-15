# Utility Function

In this folder there are tree script which add additional and separate functionality to the application. 

## Analysis

This script give to the user the opportunity to build a reward hit-map manually. Using terminal, the user uses the keyboard commands to move the agent within the selected environment. The commands to use are:

- ↑ : move forward

- → : turn rigth

- ← : turn left

From InverseReinforcementLearningApp folder run:

```bash
# example of command
python Utility/analysis.py --env-name=name of environment --reward=path/to reward model weight/file.pth
```

## Testing

This script is usefull to understand if the policy model is trained correctly. The user has to specify the environment and the policy weigth path before launching the script with terminal.

From InverseReinforcementLearningApp folder run:

```bash
# example of command
python Utility/testing.py --env-name=name of environment --policy=path/to policy weight/file.pth
```

## Training
 
This usefull script is used to understand if the reward model structure or the choosen hyperparameters are good or not. This script represent the entire algorithm described in the [paper](http://papers.nips.cc/paper/8025-reward-learning-from-human-preferences-and-demonstrations-in-atari) using a [oralce](../ReinforcementLearning/ReinforcemenLearning.md) to make the preferencies. The user can open this script and modify the hyperparameters inside the script. The main idea is that the user can modify the policy, reward model and the oracle and with this script can test faster the new model. 
For now this script can be used to test models in the `MiniGrid-Empty-6x6-v0` environment.

