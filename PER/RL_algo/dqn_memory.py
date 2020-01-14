import torch
import torch.nn as nn
import numpy as np
from collections import deque


class Net(nn.Module):
    def __init__(self, in_c, out_c):
        super(Net, self).__init__()
        c = in_c
        layers = []
        for h in [128, 64]:
            layers.append(nn.Linear(c, h))
            layers[-1].weight.data.normal_(0, 0.1)  # initialization
            # nn.init.xavier_uniform(layers[-1].weight, gain=1)
            layers.append(nn.ReLU(inplace=True))
            c = h
        layers.append(nn.Linear(c, out_c))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        actions_value = self.layers(x)
        return actions_value


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        sample_index = np.random.choice(self.capacity, batch_size)
        state, action, reward, next_state = zip(*(self.buffer[i] for i in sample_index))
        return np.concatenate(state), action, reward, np.concatenate(next_state)

    def __len__(self):
        return len(self.buffer)


class DQN(object):
    def __init__(self, state_size, action_size, buffer_size=2000,
                 exploration=0.5, lr=0.01, batch_size=32, gamma=0.9,):
        self.eval_net, self.target_net = Net(state_size, action_size),\
                                         Net(state_size, action_size)
        self.exploration = exploration
        self.action_size = action_size
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self._target_replace_iter = 100
        self.gamma = gamma  # reward decay

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = ReplayBuffer(buffer_size)  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=lr, momentum=0.95)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.rand() > self.exploration:  # greedy
            actions_value = self.eval_net.forward(x)
            # print(actions_value)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:   # random
            action = np.random.randint(0, self.action_size)
            action = action
        return action

    def store_transition(self, s, a, r, s_):
        self.memory.push(s, a, r, s_)
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter < self.buffer_size:
            return

        # target parameter update
        if self.learn_step_counter % self._target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        b_s, b_a, b_r, b_s_ = self.memory.sample(self.batch_size)
        b_s = torch.FloatTensor(b_s)
        b_a = torch.LongTensor(b_a).unsqueeze(1)
        b_r = torch.FloatTensor(b_r).unsqueeze(1)
        b_s_ = torch.FloatTensor(b_s_)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # back_propogation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.exploration > 0.1:
            self.exploration *= 0.9999994

    def load(self, name):
        states = torch.load(name)
        self.eval_net.load_state_dict(states['eval_net'])
        self.target_net.load_state_dict(states['target_net'])

    def save(self, name):
        dict_to_save = {
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict()
        }
        torch.save(dict_to_save, name)
