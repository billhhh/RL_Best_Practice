"""
Bill Wang
Oct-2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

# hyper-params
H_NEURONS = 128
LR_GAMMA = 0.99999


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, H_NEURONS)
        self.action_head = nn.Linear(H_NEURONS, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        act_probs = F.softmax(self.action_head(x), dim=-1)
        return act_probs


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, H_NEURONS)
        self.state_value = nn.Linear(H_NEURONS, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class A2C:
    gamma = 0.9
    max_grad_norm = 0.5
    A_LR = 0.001
    C_LR = 0.01

    def __init__(self, state_size, action_size, is_tensorboard=False):

        self.actor_net = Actor(state_size, action_size)
        self.critic_net = Critic(state_size)
        self.training_step = 0
        self.writer = None
        if is_tensorboard:
            self.writer = SummaryWriter('./exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.A_LR)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.C_LR)

        self.log_prob = None

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        act_probs = self.actor_net(x)
        m = Categorical(act_probs)
        action = m.sample()
        self.log_prob = m.log_prob(action)
        return action.item()

    def learn(self, s, a, r, s_):
        state = torch.FloatTensor(s)
        reward = torch.tensor(r).float()
        next_state = torch.FloatTensor(s_)

        v = self.critic_net(state)
        v_ = self.critic_net(next_state)
        target_v = reward + self.gamma * v_
        td_error = (target_v - v).detach()

        # update actor
        actor_loss = -(self.log_prob * td_error).mean()  # MAX->MIN desent
        if self.writer:
            self.writer.add_scalar('loss/actor_loss', actor_loss, global_step=self.training_step)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic
        critic_loss = F.mse_loss(v, target_v)
        if self.writer:
            self.writer.add_scalar('loss/critic_loss', critic_loss, global_step=self.training_step)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.training_step += 1

        if self.training_step % 20 == 0:
            self.A_LR *= LR_GAMMA
            self.C_LR *= LR_GAMMA
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.A_LR
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = self.C_LR
        return

    # used for multi agents
    def returnSaveDict(self):
        dict_to_save = {
            'actor_net': self.actor_net.state_dict(),
            'critic_net': self.critic_net.state_dict()
        }
        return dict_to_save

    def loadModel(self, dict_to_load):
        self.actor_net.load_state_dict(dict_to_load['actor_net'])
        self.critic_net.load_state_dict(dict_to_load['critic_net'])
