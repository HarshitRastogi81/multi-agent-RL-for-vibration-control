import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import copy

def init_weights(m):
    if isinstance(m, nn.Linear):
        fan_in = m.weight.size(1)
        bound = 1 / torch.sqrt(torch.FloatTensor([fan_in])).item()
        torch.nn.init.uniform_(m.weight, -bound, bound)
        if m.weight.size(0) == 1:
            torch.nn.init.uniform_(m.weight, -3e-3, 3e-3)
            torch.nn.init.constant_(m.bias, 3e-4)
        else:
            torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64,32)
        self.layer_4 = nn.Linear(32, action_dim)
        self.apply(init_weights)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = self.max_action * torch.tanh(self.layer_4(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim+action_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64,32)
        self.layer_4 = nn.Linear(32, action_dim)
        self.apply(init_weights)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = F.relu(self.layer_3(x1))
        x1 = self.layer_4(x1)
        return x1
    
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = deque(maxlen=int(max_size))

    def add(self, transition):
        self.storage.append(transition)

    def sample(self, batch_size):
        return random.sample(self.storage, batch_size)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=50):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dimension) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dimension) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X) + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=1e-3)
    
        self.replay_buffer = ReplayBuffer()
        self.max_action = max_action
        self.noise = OrnsteinUhlenbeckActionNoise(action_dimension=action_dim)

    # Gaussian noise:
    def select_action(self, state, noise=10):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(-self.max_action, self.max_action)
        return action
        
    # OU noise:        
    # def select_action(self, state, noise_scale=0.3):
    #     state = torch.Tensor(state.reshape(1, -1)).to(self.device)
    #     action = self.actor(state).cpu().data.numpy().flatten()
    #     noise = self.noise.sample() * noise_scale
    #     action = (action + noise).clip(-self.max_action, self.max_action)
    #     return action
    
    def reset_noise(self, sigma):
        self.noise.reset()
        self.noise.sigma = sigma
    
    def train(self, batch_size=100, gamma=0.99, tau=0.005):
        
        batch_samples = self.replay_buffer.sample(batch_size)
        state, action, next_state, reward, not_done = zip(*batch_samples)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        not_done = torch.FloatTensor(not_done).to(self.device).unsqueeze(1)

        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * gamma * target_Q).detach()

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        