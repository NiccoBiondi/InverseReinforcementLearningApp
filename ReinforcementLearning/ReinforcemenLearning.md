# Reinforcement Learning

In this folder are contained the Reinforcement Learnig  element used by the application.


## Policy

The policy model is a simple model taken from [Andrew Bagdanov repository](https://gitlab.com/bagdanov/pg-minigrid). For detailed informations please see his repository.

## Reward Model

The reward model is a simple model Which structure is similar to the policy structure.
The reward model has to understand the right reward to give to the agent .

## Oralce

The oracle is a simple class that force the reward model to learn a specific path which the agent has to do inside the `MiniGrid-Empty-6x6-v0` environment.