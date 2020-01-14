import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: torch.autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else torch.autograd.Variable(*args, **kwargs)

class DQN(nn.Module):
    def __init__(self, in_c, out_c):
        super(DQN, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        # first CNN RELU
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_c, 32, kernel_size=8,stride=4, padding=0),
            nn.BatchNorm2d(32),
            #nn.ReLU()
        )

        # second CNN RELU
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            #nn.ReLU()
        )

        # third CNN RELU
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            #nn.ReLU()
        )

        # # first FC
        # self.fc1 = nn.Sequential(
        #     nn.Linear(4*4*64, 512),
        #     nn.ReLU())

        # final FC
        self.fc2 = nn.Linear(4*4*64, self.out_c)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        # out = self.fc1(out)
        actions_value = self.fc2(out)
        return actions_value

    def act(self, env, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)))
            state = state.unsqueeze(0)
            state = state.transpose(1, 3)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
    target_model.eval()


def compute_td_loss(replay_buffer, batch_size, current_model, target_model, gamma, optimizer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    state = state.transpose(1, 3)
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    next_state = next_state.transpose(1, 3)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = current_model(state)
    next_q_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

def save(name, model):
    torch.save(model.state_dict(), name)






