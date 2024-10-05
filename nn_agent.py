import gymnasium as gym
import car_gym
from utils import SMALL_ENV, LARGE_ENV
from utils import plot_rewards
import car_gym
from car_gym.envs.hybrid_car.car import generate_random_map
from collections import namedtuple, deque
from itertools import count
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
import random

CAPACITY = 10000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 3000
LR = 1e-3
INPUT_DIM = 1600
NUM_EPISODES = 800


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
            if values[i, j]!=0:
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

env = gym.make('HybridCar-v1', desc = generate_random_map(20, 0.95, 0),  render_mode='rgb_array')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_one_hot(state):
        num_range = 1600
        one_hot_matrix = torch.eye(num_range)
        one_hot_vec = one_hot_matrix[state]
        return one_hot_vec

def get_state(observation, grid_size):
    return observation // grid_size, observation % grid_size

class DQN(nn.Module):

    def __init__(self, INPUT_DIM, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(INPUT_DIM, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)  



    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

replay_buffer = deque(maxlen=CAPACITY)
reward_buffer = deque([0.0], maxlen=100)
n_actions = env.action_space.n
state, info = env.reset()


policy_net = DQN(INPUT_DIM, n_actions).to(device)
target_net = DQN(INPUT_DIM, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
state, info = env.reset()
start_state = []
episode_reward_total = []
for episode in tqdm(range(NUM_EPISODES)):
    episode_reward = 0
    state, info = env.reset()
    for step in count():
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * step / EPS_DECAY)
        if random.random() > eps_threshold:
            action  = env.action_space.sample()
            action = torch.tensor(action, device = device)
        else:   
            one_hot_state =  get_one_hot(state)
            one_hot_state = one_hot_state.to(device)
            action = torch.argmax(policy_net(one_hot_state))

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        one_hot_next_state = get_one_hot(next_state).to(device)
        one_hot_state = get_one_hot(state)
        one_hot_state = one_hot_state.to(device)
        episode_reward+=reward
        reward = torch.tensor([reward], device = device)
        transition = (one_hot_state, action, reward, terminated, one_hot_next_state)
        replay_buffer.append(transition)
        state = next_state

        if len(replay_buffer)<BATCH_SIZE:
            continue
        else:
            transitions = random.sample(replay_buffer, BATCH_SIZE)
            states_batch = torch.stack([t[0] for t in transitions])
            action_batch = torch.stack([t[1] for t in transitions])
            reward_batch = torch.stack([t[2] for t in transitions])
            terminated_batch = np.asarray([t[3] for t in transitions])
            next_states_batch = torch.stack([t[4] for t in transitions])
            next_states_batch_non_final = []


            terminated_batch_tensor = torch.as_tensor(terminated_batch, dtype = torch.float32)
            terminated_batch_tensor = terminated_batch_tensor.unsqueeze(1)
            terminated_batch_tensor =terminated_batch_tensor.to(device)
            non_final_mask = 1-terminated_batch_tensor
            non_final_mask = non_final_mask.bool()
            for i in range(len(terminated_batch_tensor)):
                if terminated_batch_tensor[i]==0:
                    next_states_batch_non_final.append(next_states_batch[i])
            next_states_batch_non_final = torch.stack(next_states_batch_non_final)
            target_q_values_1 = torch.zeros(BATCH_SIZE,1, device=device)
            with torch.no_grad():
                target_q_values_1[non_final_mask] = target_net(next_states_batch_non_final).max(1).values
            targets = reward_batch + GAMMA*target_q_values_1
            action_batch =action_batch.unsqueeze(1)
            q_values = policy_net(states_batch)
            action_q_values = torch.gather(input = q_values, dim = 1, index = action_batch)
            criterion = nn.SmoothL1Loss()
            loss = criterion(action_q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if terminated:
            state, info = env.reset()
            reward_buffer.append(episode_reward)
            print(f"reward for episode {episode} is {episode_reward}")
            break
           
         
    get_one_hot_state_plot = get_one_hot(state)
    get_one_hot_state_plot = get_one_hot_state_plot.to(device) 
    q_init = policy_net(get_one_hot_state_plot)
    max_value_init, index = torch.max(q_values.view(-1), 0)
    print(max_value_init)
    start_state.append(max_value_init.item()) 
    episode_reward_total.append(episode_reward)
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]
    target_net.load_state_dict(target_net_state_dict)
    checkpoint_path = 'policy-net-large-1.pth'
    torch.save(policy_net.state_dict(), checkpoint_path)


epsiodes_total = np.arange(NUM_EPISODES)
plt.figure()
plt.plot(epsiodes_total,start_state, label='inital values vs episodes')
plt.xlabel('episodes')
plt.ylabel('start state')
plt.savefig('episodes_vs_start_state_plot_large_dqn.png')
plt.legend()

plt.figure()
plt.plot(epsiodes_total,episode_reward_total, label = 'total reward vs episodes')
plt.xlabel('episdoes')
plt.ylabel('discounted rewards')
plt.legend()
plt.savefig('reward_vs_episode_large_dqn.png')



with torch.no_grad():
    policy_net.load_state_dict(torch.load('policy-net-large.pth'))
    for i in tqdm(range(1)):
        state, info = env.reset()
        values = np.zeros((20,20))
        actions = np.zeros((20,20))
        m,n = get_state(state,20)
        one_hot_state = get_one_hot(state)
        one_hot_state = one_hot_state.to(device)
        q_values = policy_net(one_hot_state)
        max_value, index = torch.max(q_values.view(-1), 0)
        action = index.item()
        values[m][n] = max_value
        actions[m][n]=action
        flag = False
        total_reward = 0
        while not flag:
            if random.random()>1:
                action = env.action_space.sample()
            else:
                one_hot_state = get_one_hot(state)
                one_hot_state = one_hot_state.to(device)
                q_values = policy_net(one_hot_state)
                max_value, index = torch.max(q_values.view(-1), 0)
                action = index.item()

            next_state, reward, terminated, truncated, info = env.step(action)
            m,n = get_state(state,40)

            values[m][n] = max_value
            actions[m][n] = action
            state = next_state
            total_reward+=reward
            if terminated == True:
                flag = True

plotting(values,actions)