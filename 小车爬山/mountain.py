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
    def __init__(self, n_states, n_actions, memory_capacity, TARGET_UPDATE_STEP, batch_size, lr, epsilon, gamma):
        self.epsilon = epsilon
        self.lr = lr
        self.n_states = n_states
        self.n_actions = n_actions
        self.memory_capacity = memory_capacity
        self.target_update_step = TARGET_UPDATE_STEP
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eval_net, self.target_net = Net(n_states, n_actions).to(self.device), Net(n_states, n_actions).to(
            self.device)
        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)  # torch 的优化器
        self.loss_func = nn.MSELoss()  # 误差公式

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)
        if np.random.uniform() < self.epsilon:  # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].cpu().numpy()[0]
        else:  # 选随机动作
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % self.target_update_step == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]).to(self.device)
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2]).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:]).to(self.device)

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

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")  # 立杆子游戏
    env = env.unwrapped
    N_ACTIONS = env.action_space.n  # 杆子能做的动作
    N_STATES = env.observation_space.shape[0]  # 杆子能获取的环境信息数
    dqn = DQN(N_STATES, N_ACTIONS, MEMORY_CAPACITY, TARGET_UPDATE_STEP, BATCH_SIZE, LR, EPSILON, GAMMA)

    for epoch in range(5000):
        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            a = dqn.choose_action(s)

            # 执行动作
            s_, r, done, info = env.step(a)

            if s_[0] > -0.4 and s_[0] < 0.5:
                r = 10 * (s_[0] + 0.4) ** 3
            elif s_[0] >= 0.5:
                r = 100
            elif s_[0] <= -0.4:
                r = -0.1

            ep_r += r

            dqn.store_transition(s, a, r, s_)
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            if done:
                print("【Epoch】: ", epoch, "【Reward】: ", round(ep_r, 2))
                break
            s = s_
