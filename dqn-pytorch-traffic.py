import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, in_c, out_c):
        super(Net, self).__init__()
        c = in_c
        layers = []
        for h in [128, 64, out_c]:
            layers.append(nn.Linear(c, h))
            layers.append(nn.ReLU(inplace=True))
            c = h
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        actions_value = self.layers(x)
        return actions_value


class DQN(object):
    def __init__(self, state_size, action_size, exploration,
                 lr=0.001, batch_size=100, gamma=0.9):
        buffer_size = 2000
        self.eval_net, self.target_net = Net(state_size, action_size),\
                                         Net(state_size, action_size)
        self.exploration = exploration
        self.action_size = action_size
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self._target_replace_iter = 100
        self.gamma = gamma

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((buffer_size, state_size * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def act(self, x):
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

    def remember(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.buffer_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def replay(self):
        if self.memory_counter < self.buffer_size:
            return

        # target parameter update
        if self.learn_step_counter % self._target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.buffer_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_size])
        b_a = torch.LongTensor(b_memory[:, self.state_size:self.state_size+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_size+1:self.state_size+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_size:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # back_propogation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.exploration > 0.01:
            self.exploration *= 0.99999

    def load(self, name):
        states = torch.load(name)
        self.eval_net.load_state_dict(states['eval_net'])
        self.target_net.load_state_dict(states['target_net'])

    def loadModel(self, dict_to_load):
        self.eval_net.load_state_dict(dict_to_load['eval_net'])
        self.target_net.load_state_dict(dict_to_load['target_net'])

    # used for single agent
    def save(self, name):
        dict_to_save = {
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict()
        }
        torch.save(dict_to_save, name)

    # used for multi agents
    def returnSaveDict(self):
        dict_to_save = {
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict()
        }
        return dict_to_save
