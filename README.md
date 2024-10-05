# Tabular Q Learning and Deep Q Learning 
This repositiory contains an implementation of Tabular Q Learning and Deep-Q-Learning for custom car environment. 

## Project Overview
We provide implementation and analyses for:
- **Tabular Q Learning**: Uses dynamic programming to calculate the goodness of each state (Q values) for a given environement without explicitly knowing the transition dynamics of the environment by using epsilon greedy exploration to find the optimal policy 
- **Deep Q Learning**- We learn Q values of all the states by using a neural network and maintaining a one hot encoding of the states

## Files Overview
- nn_agent.py contains implementation of deep Q learning for a custom car -environment which can be modified suitably for any environment

- tabular_agent.py contains implementation of Tabular Q learning for custom car environment which can be modified suitably for any environment

## Report
- Implementation details of both the algorithms
- Analyses and impact of hyperparameters and experiments conducted
