import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_actions):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)  # output(n,8,160,140)
        self.pool1 = nn.MaxPool2d(2, 2)  # output(n, 8, 80, 70)
        self.conv2 = nn.Conv2d(8, 32, 5)  # output(n,32, 76, 66)
        self.pool2 = nn.MaxPool2d(2, 2)  # output(n,32,38,33)
        self.conv3 = nn.Conv2d(32, 16, 3)  # output(n,16, 36, 31)
        self.pool3 = nn.MaxPool2d(2, 2)  # output(n,16,18,15)
        self.fc1 = nn.Linear(16 * 18 * 15, 32)
        self.fc2 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.fc1(x.view(x.shape[0], -1))
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DQN(object):
    def __init__(self, states, n_actions, memory_capacity, target_update_step, batch_size, lr, epsilon, gamma):
        self.epsilon = epsilon
        self.lr = lr
        self.states = states
        self.n_actions = n_actions
        self.memory_capacity = memory_capacity
        self.target_update_step = target_update_step
        self.batch_size = batch_size
        self.gamma = gamma

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eval_net, self.target_net = Net(n_actions).to(self.device), Net(n_actions).to(self.device)
        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory_s = np.zeros((self.memory_capacity, states[2], states[0], states[1]))
        self.memory_a = np.zeros((self.memory_capacity, 1))
        self.memory_r = np.zeros(self.memory_capacity)
        self.memory_s_ = np.zeros((self.memory_capacity, states[2], states[0], states[1]))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)  # torch 的优化器
        self.loss_func = nn.MSELoss()  # 误差公式

    def choose_action(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        if np.random.uniform() < self.epsilon:  # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].cpu().numpy()[0]
        else:  # 选随机动作
            action = np.random.randint(0, self.n_actions)
        return action

    def store(self, s, a, r, s_):
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.memory_capacity
        self.memory_s[index] = s
        self.memory_a[index] = [a]
        self.memory_r[index] = r
        self.memory_s_[index] = s_
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % self.target_update_step == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_s = self.memory_s[sample_index]
        b_s = torch.FloatTensor(b_s).to(self.device)
        b_a = self.memory_a[sample_index]
        b_a = torch.LongTensor(b_a.astype(int)).to(self.device)
        b_r = self.memory_r[sample_index]
        b_r = torch.FloatTensor(b_r).to(self.device)
        b_s_ = self.memory_s_[sample_index]
        b_s_ = torch.FloatTensor(b_s_).to(self.device)

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r.view(self.batch_size, 1) + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 超参数
BATCH_SIZE = 32
LR = 0.001  # 学习率
EPSILON = 0.9  # 最优选择动作百分比
GAMMA = 0.99  # 奖励递减参数
TARGET_UPDATE_STEP = 100  # Q 现实网络的更新频率
MEMORY_CAPACITY = 10000  # 记忆库大小
EPISODE = 100000
EPISODE_STEP = 2000
LEARN_START = 1000

if __name__ == "__main__":
    env = gym.make("Breakout-v4")
    env = env.unwrapped
    N_ACTIONS = env.action_space.n  # 动作数量
    H, W, C = env.observation_space.shape  # 环境信息维度
    dqn = DQN((H - 46, W - 16, 1), N_ACTIONS, MEMORY_CAPACITY, TARGET_UPDATE_STEP, BATCH_SIZE, LR, EPSILON, GAMMA)

    e_reward = 0
    for episode in range(1, EPISODE + 1):
        s = env.reset()
        s = s[32:-14, 8:-8, 0, None]
        s = s.transpose(2, 0, 1)
        s[s > 0] = 1
        total_reward = 0
        for step in range(EPISODE_STEP):
            env.render()
            a = dqn.choose_action(s)

            # 执行动作
            s_, r, done, info = env.step(a)
            s_ = s_[32:-14, 8:-8, 0, None]
            s_ = s_.transpose(2, 0, 1)
            s_[s_ > 0] = 1
            dqn.store(s, a, r, s_)

            total_reward += r
            if dqn.memory_counter > LEARN_START:
                dqn.learn()

            if done:
                break
            s = s_
        print("episode: ", episode, "Reward: ", total_reward)
        e_reward += total_reward
        if episode % 10 == 0:
            print("total_reward/10: ", e_reward / 10)
            e_reward = 0
