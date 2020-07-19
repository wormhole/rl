import os
import random
import sys
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

sys.path.append("game/")
import wrapped_flappy_bird as game

actions = 2
gamma = 0.99
observe = 1000
explore = 2000000
final_epsilon = 0.001
initial_epsilon = 0.4
memory_capacity = 10000
batch_size = 32
frame_per_action = 4
target_update_step = 100
lr = 1e-6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir="log", comment="dqn")


def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80))


class QNet(nn.Module):
    def __init__(self, ):
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
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.out(x)


class DQNAgent(object):
    def __init__(self):
        self.replay_memory = deque()
        self.time_step = 0
        self.epsilon = initial_epsilon
        self.q_net = QNet().to(device)
        self.target_q_net = QNet().to(device)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def save(self):
        torch.save(self.q_net.state_dict(), "params.pth")

    def load(self):
        if os.path.exists("params.pth"):
            self.q_net.load_state_dict(torch.load("params.pth"))
            self.target_q_net.load_state_dict(torch.load("params.pth"))

    def learn(self):
        batch = random.sample(self.replay_memory, batch_size)
        state_batch = [data[0] for data in batch]
        action_batch = [data[1] for data in batch]
        reward_batch = [data[2] for data in batch]
        _state_batch = [data[3] for data in batch]

        _state_batch = np.array(_state_batch)
        _state_batch = torch.Tensor(_state_batch).to(device)
        action_batch = np.array(action_batch)
        index = action_batch.argmax(axis=1)
        index = np.reshape(index, [batch_size, 1])
        action_index_batch = torch.LongTensor(index).to(device)
        target_q_value = self.target_q_net(_state_batch)
        target_q_value = target_q_value.detach().cpu().numpy()

        y_batch = np.zeros([batch_size, 1])
        for i in range(0, batch_size):
            if batch[i][4]:
                y_batch[i][0] = reward_batch[i]
            else:
                y_batch[i][0] = reward_batch[i] + gamma * np.max(target_q_value[i])

        y_batch = np.array(y_batch)
        y_batch = np.reshape(y_batch, [batch_size, 1])
        state_batch = torch.Tensor(state_batch).to(device)
        y_batch = torch.Tensor(y_batch).to(device)
        y_predict = self.q_net(state_batch).gather(1, action_index_batch)
        loss = self.loss_func(y_predict, y_batch)
        writer.add_scalar("loss", loss.item(), self.time_step)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.time_step % target_update_step == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.save()

    def store(self, state, action, reward, _state, terminal):
        self.time_step += 1
        self.replay_memory.append((state, action, reward, _state, terminal))
        if len(self.replay_memory) > memory_capacity:
            self.replay_memory.popleft()
        if self.time_step > observe:
            self.learn()
        writer.add_scalar("reward", reward, self.time_step)

    def choose_action(self, state):
        if self.epsilon != 1 and self.epsilon > final_epsilon and self.time_step > observe and (
                self.time_step - observe) % 1000 == 0:
            self.epsilon -= (initial_epsilon - final_epsilon) * 1000 / explore

        action = np.zeros(actions)
        if self.time_step % frame_per_action == 0:
            if self.epsilon != 1 and random.random() <= self.epsilon:
                action_index = random.randrange(actions)
                action[action_index] = 1
            else:
                state = torch.Tensor([state]).to(device)
                q_value = self.q_net(state)
                action_index = q_value.detach().cpu().max(1)[1].item()
                action[action_index] = 1
        else:
            action[0] = 1
        return action


def train(env, agent):
    total_reward = 0
    action = np.array([1, 0])
    state, reward, terminal = env.frame_step(action)
    state = preprocess(state)
    state = np.stack((state, state, state, state), axis=0)

    episode = 1

    while True:
        action = agent.choose_action(state)
        _state, reward, terminal = env.frame_step(action)
        _state = preprocess(_state)
        _state = np.append(state[1:, :, :], np.expand_dims(_state, axis=0), axis=0)
        agent.store(state, action, reward, _state, terminal)
        total_reward += reward
        if terminal:
            print("episode: ", episode, "step: ", agent.time_step, "reward: ", total_reward, "epsilon: ", agent.epsilon)
            total_reward = 0
            episode += 1
            action = np.array([1, 0])
            _state, reward, terminal = env.frame_step(action)
            _state = preprocess(_state)
            _state = np.stack((_state, _state, _state, _state), axis=0)
        state = _state


def test(env, agent):
    agent.load()
    agent.epsilon = 1
    total_reward = 0
    action = np.array([1, 0])
    state, reward, terminal = env.frame_step(action)
    state = preprocess(state)
    state = np.stack((state, state, state, state), axis=0)

    episode = 1
    while True:
        action = agent.choose_action(state)
        _state, reward, terminal = env.frame_step(action)
        _state = preprocess(_state)
        _state = np.append(state[1:, :, :], np.expand_dims(_state, axis=0), axis=0)
        total_reward += reward
        if terminal:
            print("episode: ", episode, "step: ", agent.time_step, "reward: ", total_reward, "epsilon: ", agent.epsilon)
            episode += 1
            total_reward = 0
            action = np.array([1, 0])
            _state, reward, terminal = env.frame_step(action)
            _state = preprocess(_state)
            _state = np.stack((_state, _state, _state, _state), axis=0)
        state = _state


if __name__ == "__main__":
    agent = DQNAgent()
    flappy_bird = game.GameState()
    train(flappy_bird, agent)
