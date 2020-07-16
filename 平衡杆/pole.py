import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, n_actions)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.fc2(x)
        return actions_value


class DQN(object):
    def __init__(self, n_states, n_actions, memory_capacity, target_update_step, batch_size, lr, epsilon, gamma):
        self.epsilon = epsilon
        self.lr = lr
        self.n_states = n_states
        self.n_actions = n_actions
        self.memory_capacity = memory_capacity
        self.target_update_step = target_update_step
        self.batch_size = batch_size
        self.gamma = gamma

        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)  # torch 的优化器
        self.loss_func = nn.MSELoss()  # 误差公式

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.epsilon:  # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:  # 选随机动作
            action = np.random.randint(0, self.n_actions)
        return action

    def store(self, s, a, r, s_):
        mem = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = mem
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % self.target_update_step == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 超参数
BATCH_SIZE = 32
LR = 0.001  # 学习率
EPSILON = 0.9  # 最优选择动作百分比
GAMMA = 0.95  # 奖励递减参数
TARGET_UPDATE_STEP = 100  # Q 现实网络的更新频率
MEMORY_CAPACITY = 10000  # 记忆库大小
EPISODE = 100000
EPISODE_STEP = 2000
LEARN_START = 1000

if __name__ == "__main__":
    env = gym.make("CartPole-v0")  # 立杆子游戏
    N_ACTIONS = env.action_space.n  # 动作数量
    N_STATES = env.observation_space.shape[0]  # 环境信息维度
    dqn = DQN(N_STATES, N_ACTIONS, MEMORY_CAPACITY, TARGET_UPDATE_STEP, BATCH_SIZE, LR, EPSILON, GAMMA)

    e_reward = 0
    for episode in range(1, EPISODE + 1):
        s = env.reset()
        total_reward = 0
        for step in range(EPISODE_STEP):
            env.render()
            a = dqn.choose_action(s)

            # 执行动作
            s_, r, done, info = env.step(a)

            # 重新计算reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

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
