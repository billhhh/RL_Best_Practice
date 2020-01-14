import gym
from utlis import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from collections import deque # Ordered collection with ends
import utlis.net_functions as nF
import torch
import math
import random
from tqdm import tqdm

#######################################################################################
# env init
env = gym.make('SpaceInvaders-v0')
env = env.unwrapped
env.seed(21)

action_size = env.action_space.n
print('Action size:', action_size)
state_size = env.observation_space
print('State size:', state_size)

#######################################################################################

# training parameters
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: torch.autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else torch.autograd.Variable(*args, **kwargs)

epsilon_start = 0.5
epsilon_final = 0.1
epsilon_decay = 50000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

num_frames = 5000000
batch_size = 32
gamma      = 0.99
stack_size = 4

losses = []
all_rewards = []
episode_reward = 0

#######################################################################################

# initial model and buffer

current_model = nF.DQN(4, action_size)
target_model = nF.DQN(4, action_size)
if USE_CUDA:
    current_model = current_model.cuda()
    target_model = target_model.cuda()

# optimizer = torch.optim.RMSprop(current_model.parameters(), lr=0.00025, momentum=0.95)
optimizer = torch.optim.Adam(current_model.parameters(), lr=0.001)

# initialize replay buffer
replay_buffer = nF.ReplayBuffer(50000)
nF.update_target(current_model, target_model)

#######################################################################################

# start training
state = env.reset()
new_episode = True

for frame_idx in tqdm(range(1, num_frames + 1)):
    current_model.train()
    epsilon = epsilon_by_frame(frame_idx)

    if frame_idx == 1:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
        state, stacked_frames = preprocessing.stack_frames(stack_size, stacked_frames, state, True)

    action = current_model.act(env, state, epsilon)

    next_state, reward, done, _ = env.step(action)
    episode_reward += reward

    if done:
        next_state = np.zeros((84, 84), dtype=np.int)
        next_state, stacked_frames = preprocessing.stack_frames(stack_size, stacked_frames, next_state, False)
        replay_buffer.push(state, action, reward, next_state, done)

        all_rewards.append(episode_reward)
        state = env.reset()
        episode_reward = 0
        state, stacked_frames = preprocessing.stack_frames(stack_size, stacked_frames, state, True)
    else:
        next_state, stacked_frames = preprocessing.stack_frames(stack_size, stacked_frames, next_state, False)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

    if len(replay_buffer) > batch_size:
        loss = nF.compute_td_loss(replay_buffer, batch_size, current_model, target_model, gamma, optimizer)
        losses.append(loss.item())

    # if frame_idx % 200 == 0:
    #     nF.plot(frame_idx, all_rewards, losses)

    if frame_idx % 100 == 0:
        nF.update_target(current_model, target_model)

    if frame_idx % 1000 == 0:
        nF.save('./models/iter_'+str(frame_idx)+'.pt', current_model)

        # start eval()
        current_model.eval()
        with torch.no_grad():
            eval_state = env.reset()
            eval_episode_reward = 0
            eval_all_rewards = []
            for eval_frame_idx in range(5000):
                env.render()
                if eval_frame_idx == 0:
                    eval_stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
                    eval_state, eval_stacked_frames = preprocessing.stack_frames(stack_size, eval_stacked_frames,
                                                                                 eval_state, True)

                eval_action = current_model.act(env, eval_state, 0.)
                eval_next_state, eval_reward, eval_done, _ = env.step(eval_action)
                eval_episode_reward += eval_reward

                if eval_done:
                    eval_all_rewards.append(eval_episode_reward)
                    eval_state = env.reset()
                    eval_episode_reward = 0
                    eval_state, eval_stacked_frames = preprocessing.stack_frames(stack_size, eval_stacked_frames, eval_state, True)
                else:
                    eval_next_state, eval_stacked_frames = preprocessing.stack_frames(stack_size, eval_stacked_frames, eval_next_state, False)
                    eval_state = eval_next_state

            print('Eval score: rewards={} mean={}, std={}, epsilon={}'.format(eval_all_rewards,
                                                                              np.mean(eval_all_rewards),
                                                                              np.std(eval_all_rewards),
                                                                              epsilon))






















