import torch
import torch.nn as nn
import numpy as np
from collections import deque


class Net(nn.Module):
    def __init__(self, in_c, out_c):
        super(Net, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_c, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        nn.init.xavier_uniform_(self.layer1[0].weight, gain=1)

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        nn.init.xavier_uniform_(self.layer2[0].weight, gain=1)

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        nn.init.xavier_uniform_(self.layer3[0].weight, gain=1)

        # first FC
        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(7*7*64, 512),
            nn.ReLU())

        nn.init.xavier_uniform_(self.fc1[1].weight, gain=1)

        # final FC
        self.fc2 = nn.Linear(512, self.out_c)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        actions_value = self.fc2(out)
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
    def __init__(self, state_size, action_size, buffer_size=500,
                 epsilon=0.5, epsilon_decay=0.99999, lr=0.0002, lr_decay=0.5,
                 lr_decay_epoch=5000, batch_size=32, gamma=0.9, use_cuda=False,
                 eval=False, target_replace_iter=10000):
        self.eval_net, self.target_net = Net(state_size, action_size),\
                                         Net(state_size, action_size)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self._target_replace_iter = target_replace_iter
        self.gamma = gamma  # reward decay
        self.use_cuda = use_cuda
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_epoch = lr_decay_epoch

        if use_cuda:
            self.eval_net.cuda()
            self.target_net.cuda()

        if eval:
            self.eval_net.eval()
            self.target_net.eval()
        else:
            self.eval_net.train()
            self.target_net.train()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = ReplayBuffer(buffer_size)     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=lr, momentum=0.95)
        self.loss_func = nn.MSELoss()
        self.ave_q_list = [0]

    def choose_action(self, x):
        if np.random.rand() > self.epsilon:  # greedy
            x = torch.FloatTensor(x).unsqueeze(0)
            x = x.permute(0, 3, 1, 2)
            if self.use_cuda:
                x = x.cuda()

            actions_value = self.eval_net.forward(x)
            # print(actions_value)
            action = actions_value.max(1)[1].data[0]
            self.ave_q_list.append(actions_value.max(1)[0].item())
        else:   # random
            action = np.random.randint(0, self.action_size)
            action = action
        return action

    def store_transition(self, s, a, r, s_):
        self.memory.push(s, a, r, s_)
        self.memory_counter += 1

    def learn(self, episode):
        if self.memory_counter < self.buffer_size:
            return

        if episode % self.lr_decay_epoch == 0 and episode > 0:
            self.adjust_learning_rate()

        # target parameter update
        if self.learn_step_counter % self._target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        b_s, b_a, b_r, b_s_ = self.memory.sample(self.batch_size)
        b_s = torch.FloatTensor(b_s)
        b_s = b_s.permute(0, 3, 1, 2)
        b_s_ = torch.FloatTensor(b_s_)
        b_s_ = b_s_.permute(0, 3, 1, 2)
        b_a = torch.LongTensor(b_a).unsqueeze(1)
        b_r = torch.FloatTensor(b_r).unsqueeze(1)

        if self.use_cuda:
            b_s = b_s.cuda()
            b_s_ = b_s_.cuda()
            b_a = b_a.cuda()
            b_r = b_r.cuda()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # back_propogation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > 0.1:
            self.epsilon *= self.epsilon_decay

    def save(self, path, episode, total_steps):
        dict_to_save = {
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'lr': self.lr,
            'optimizer': self.optimizer,
            'epsilon': self.epsilon,
            'episode': episode,
            'total_steps': total_steps,
        }
        torch.save(dict_to_save, path)

    def load(self, path):
        states = torch.load(path)
        self.eval_net.load_state_dict(states['eval_net'])
        self.target_net.load_state_dict(states['target_net'])
        self.lr = states['lr']
        self.optimizer = states['optimizer']
        self.epsilon = states['epsilon']
        episode = states['episode']
        total_steps = states['total_steps']
        print("Resume episode ==> ", episode)
        print("Resume steps ==> ", total_steps)
        return episode, total_steps

    def get_epsilon(self):
        return self.epsilon

    def get_lr(self):
        return self.lr

    def adjust_learning_rate(self):
        self.lr *= self.lr_decay
        print(' * adjust lr == {}'.format(self.lr))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def get_ave_q(self):
        return np.mean(self.ave_q_list)

    def episode_done(self):  # something after episode done
        self.ave_q_list = [0]
