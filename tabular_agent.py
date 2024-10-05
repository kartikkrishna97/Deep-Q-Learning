import gymnasium as gym
import car_gym
from car_gym.envs.hybrid_car.car import generate_random_map
# from utils import SMALL_ENV, LARGE_ENV
from utils import plot_rewards
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import random


def plotting(values, actions):


    
    arrow_directions = {
        0: (0, -1),   
        1: (-1, 0),    
        2: (0, 1),    
        3: (1, 0)    
    }


    plt.figure(figsize=(8, 8))
    heatmap = plt.imshow(values, cmap='viridis', interpolation='nearest')

    
    for i in range(actions.shape[0]):
        for j in range(actions.shape[1]):
            action = actions[i, j]
            arrow_direction = arrow_directions[action]
            if value[i, j]!=0:
                plt.quiver(j, i, arrow_direction[1], arrow_direction[0], scale=100, color='black')

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if values[i, j] == 0:
                plt.text(j, i, 'T', ha='center', va='center', color='black', fontsize=8)
            else:
                plt.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', color='black', fontsize=4)

    cbar = plt.colorbar(heatmap)
    cbar.set_label('Values')

    plt.title('Q-values with actions')

    plt.savefig('Tabular-Large.png')


def get_state(observation, grid_size):
    return observation // grid_size, observation % grid_size

NUM_EPISODES = 5000
ALPHA = 0.3
gamma =  1
epsilon = 0.2
env =  gym.make('HybridCar-v1', desc = generate_random_map(40, 0.95, 0),  render_mode='rgb_array')



Q_init =  np.zeros((env.observation_space.n,env.action_space.n)) 

start_state = []
for _ in range(NUM_EPISODES):
    state, info = env.reset()
    flag =  False
    while not flag:
        if random.random() < epsilon:
            action = env.action_space.sample()

        else:
            action = np.argmax(Q_init[state])
        
        next_state, reward, terminated, truncated, info = env.step(action)
        target = reward + gamma* np.max(Q_init[next_state])
        Q_init[state][action] = Q_init[state][action]  + ALPHA*(target-Q_init[state][action])
        state = next_state

        if terminated == True:
            flag = True
    start_state.append(np.max(Q_init[0]))
        
total_episodes = np.arange(NUM_EPISODES)
start_state = np.array(start_state)



plt.figure()
plt.plot(total_episodes, start_state)
plt.title('Start State vs Total Episodes')
plt.xlabel('Total Episodes')
plt.ylabel('Start State')
plt.savefig('start_state_vs_episodes_large.png')

value =  np.zeros((40, 40))
actions = np.zeros((40,40))
for i in range(len(Q_init)):
    m,n = get_state(i,40)
    value_max = np.max(Q_init[i])
    value[m][n] = value_max
    actions[m][n] = np.argmax(Q_init[i])



print(actions)
plotting(value,actions)
