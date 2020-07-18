import os
import random
import time
from collections import deque

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

# 超参数
batch_size = 32
lr = 1e-6
epsilon = 0.1
gamma = 0.99
target_update_step = 100
memory_capacity = 10000
episode = 20000
episode_step = 2000
observe = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir="log", comment="dqn")


def preprocess(state):
    state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(state, (80, 80))


class QNet(nn.Module):
    def __init__(self, n_actions):
        super(QNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU()
        )
        self.out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.out(x)
        return x


class DQNAgent(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.time_step = 0
        self.replay_memory = deque()
        self.epsilon = epsilon

        self.q_net = QNet(n_actions).to(device)
        self.target_q_net = QNet(n_actions).to(device)
        # writer.add_graph(self.q_net, (torch.randn(32, 4, 80, 80),))

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def save(self):
        torch.save(self.q_net.state_dict(), "params.pth")

    def load(self):
        if os.path.exists("params.pth"):
            self.q_net.load_state_dict(torch.load("params.pth"))
            self.target_q_net.load_state_dict(torch.load("params.pth"))

    def choose_action(self, state):
        if self.epsilon == 1 or np.random.uniform() > self.epsilon:
            state = torch.FloatTensor([state]).to(device)
            q_value = self.q_net.forward(state)
            action = torch.max(q_value, 1)[1].cpu().numpy()[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store(self, state, action, reward, _state):
        self.time_step += 1
        self.replay_memory.append((state, action, reward, _state))
        if len(self.replay_memory) > memory_capacity:
            self.replay_memory.popleft()

        if self.time_step > observe:
            self.learn()

    def learn(self):
        batch = random.sample(self.replay_memory, batch_size)
        state_batch = [data[0] for data in batch]
        action_batch = [data[1] for data in batch]
        reward_batch = [data[2] for data in batch]
        _state_batch = [data[3] for data in batch]

        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.LongTensor(action_batch).view(batch_size, 1).to(device)
        reward_batch = torch.FloatTensor(reward_batch).view(batch_size, 1).to(device)
        _state_batch = torch.FloatTensor(_state_batch).to(device)

        q_value = self.q_net(state_batch).gather(1, action_batch)
        target_q_value = self.target_q_net(_state_batch).detach()
        target_q_value = target_q_value.max(1)[0].view(batch_size, 1)
        target_q_value = reward_batch + gamma * target_q_value

        loss = self.loss_func(q_value, target_q_value)
        writer.add_scalar("loss", loss.item(), self.time_step)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.time_step % target_update_step == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.save()


def train(env, agent):
    total_reward = 0

    for ep in range(1, episode + 1):
        env.reset()
        ep_reward = 0

        state, reward, done, info = env.step(1)
        alive = info["ale.lives"]
        state = preprocess(state)
        state = np.stack((state, state, state, state), axis=0)

        for step in range(episode_step):
            env.render()
            action = agent.choose_action(state)

            _state, reward, done, info = env.step(action)
            _state = preprocess(_state)
            _state = np.append(state[1:, :, :], np.expand_dims(_state, axis=0), axis=0)
            agent.store(state, action, reward, _state)

            ep_reward += reward

            if done:
                break
            elif alive != info["ale.lives"]:
                _state, reward, done, info = env.step(1)
                alive = info["ale.lives"]
                _state = preprocess(_state)
                _state = np.stack((_state, _state, _state, _state), axis=0)
            state = _state

        print("episode: ", ep, "reward: ", ep_reward)
        total_reward += ep_reward
        if ep % 10 == 0:
            print("av_reward(10): ", total_reward / 10)
            total_reward = 0


def test(env, agent):
    agent.load()
    agent.epsilon = 1
    total_reward = 0

    for ep in range(1, episode + 1):
        env.reset()
        ep_reward = 0

        state, reward, done, info = env.step(1)
        alive = info["ale.lives"]
        state = preprocess(state)
        state = np.stack((state, state, state, state), axis=0)

        for step in range(episode_step):
            time.sleep(0.05)
            env.render()
            action = agent.choose_action(state)

            _state, reward, done, info = env.step(action)
            _state = preprocess(_state)
            _state = np.append(state[1:, :, :], np.expand_dims(_state, axis=0), axis=0)

            ep_reward += reward

            if done:
                break
            elif alive != info["ale.lives"]:
                _state, reward, done, info = env.step(1)
                alive = info["ale.lives"]
                _state = preprocess(_state)
                _state = np.stack((_state, _state, _state, _state), axis=0)
            state = _state

        print("episode: ", ep, "reward: ", ep_reward)
        total_reward += ep_reward
        if ep % 10 == 0:
            print("av_reward(10): ", total_reward / 10)
            total_reward = 0


if __name__ == "__main__":
    env = gym.make("Breakout-v4")
    n_actions = env.action_space.n
    agent = DQNAgent(n_actions)
    train(env, agent)
    # test(env, agent)
