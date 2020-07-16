import os
import random
import sys
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.append("game/")
import wrapped_flappy_bird as game

GAME = "bird"  # 游戏名
ACTIONS = 2  # 动作数量
GAMMA = 0.99  # 奖励折扣率
OBSERVE = 1000  # 多少次后进行训练
EXPLORE = 2000000  # 探索次数
FINAL_EPSILON = 0.0001  # 最终epsilon
INITIAL_EPSILON = 0.1  # 初始epsilon
REPLAY_MEMORY = 50000  # 记忆回放容量
BATCH_SIZE = 32
FRAME_PER_ACTION = 1
UPDATE_TIME = 100  # 多少次后更新target_q_net
width = 80
height = 80
LR = 1e-6  # 学习率
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80))


class DeepNetWork(nn.Module):
    def __init__(self, ):
        super(DeepNetWork, self).__init__()
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
    def __init__(self, actions):
        self.replay_memory = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        self.q_net = DeepNetWork().to(device)
        self.target_q_net = DeepNetWork().to(device)
        self.load()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)

    def save(self):
        torch.save(self.q_net.state_dict(), "params3.pth")

    def load(self):
        if os.path.exists("params3.pth"):
            self.q_net.load_state_dict(torch.load("params3.pth")).to(device)
            self.target_q_net.load_state_dict(torch.load("params3.pth")).to(device)

    def train(self):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in batch]
        action_batch = [data[1] for data in batch]
        reward_batch = [data[2] for data in batch]
        _state_batch = [data[3] for data in batch]

        _state_batch = np.array(_state_batch)
        _state_batch = torch.Tensor(_state_batch).to(device)
        action_batch = np.array(action_batch)
        index = action_batch.argmax(axis=1)
        index = np.reshape(index, [BATCH_SIZE, 1])
        action_index_batch = torch.LongTensor(index).to(device)
        q_value_batch = self.target_q_net(_state_batch)
        q_value_batch = q_value_batch.detach().cpu().numpy()

        y_batch = np.zeros([BATCH_SIZE, 1])
        for i in range(0, BATCH_SIZE):
            terminal = batch[i][4]
            if terminal:
                y_batch[i][0] = reward_batch[i]
            else:
                # 这里的q_value_batch[i]为数组，大小为所有动作集合大小，q_value_batch[i],代表
                # 做所有动作的Q值数组，y计算为如果游戏停止，y=reward[i],如果没停止，则y=reward[i]+gamma*np.max(q_value[i])
                # 代表当前y值为当前reward+未来预期最大值*gamma(gamma:经验系数)
                y_batch[i][0] = reward_batch[i] + GAMMA * np.max(q_value_batch[i])

        y_batch = np.array(y_batch)
        y_batch = np.reshape(y_batch, [BATCH_SIZE, 1])
        state_batch = torch.Tensor(state_batch).to(device)
        y_batch = torch.Tensor(y_batch).to(device)
        y_predict = self.q_net(state_batch).gather(1, action_index_batch)
        loss = self.loss_func(y_predict, y_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.time_step % UPDATE_TIME == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.save()

    def set_perception(self, next_observation, action, reward, terminal):
        _state = np.append(self.state[1:, :, :], np.expand_dims(next_observation, axis=0), axis=0)
        self.replay_memory.append((self.state, action, reward, _state, terminal))
        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()
        if self.time_step > OBSERVE:
            self.train()

        status = ""
        if self.time_step <= OBSERVE:
            status = "observe"
        elif self.time_step > OBSERVE and self.time_step <= OBSERVE + EXPLORE:
            status = "explore"
        else:
            status = "train"
        print("TIME_STEP", self.time_step, "/ STATUS", status, "/ EPSILON", self.epsilon)
        self.state = _state
        self.time_step += 1

    def get_action(self):
        current_state = torch.Tensor(self.state).unsqueeze(dim=0).to(device)
        q_value = self.q_net(current_state)[0]
        action = np.zeros(self.actions)
        if self.time_step % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(q_value.detach().cpu().numpy())
                action[action_index] = 1
        else:
            action[0] = 1

        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return action

    def init_state(self, observation):
        self.state = np.stack((observation, observation, observation, observation), axis=0)


if __name__ == "__main__":
    actions = 2
    agent = DQNAgent(actions)
    flappy_bird = game.GameState()

    action0 = np.array([1, 0])
    observation0, reward0, terminal = flappy_bird.frame_step(action0)
    observation0 = preprocess(observation0)

    agent.init_state(observation0)

    while True:
        action = agent.get_action()
        next_observation, reward, terminal = flappy_bird.frame_step(action)
        next_observation = preprocess(next_observation)
        agent.set_perception(next_observation, action, reward, terminal)
