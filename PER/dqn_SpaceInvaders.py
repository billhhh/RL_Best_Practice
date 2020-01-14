import gym
from RL_algo.dqn_cnn import DQN
from RL_algo.ddqn_cnn import DDQN
import numpy as np
from skimage.color import rgb2gray
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt
import time
import random
import torch
import os


# env init
env = gym.make('SpaceInvaders-v0')
env = env.unwrapped
skip_learning = 1
env.frameskip = 3
env.seed(21)
MEMORY_SIZE = 100000
USE_CUDA = True
stack_size = 4
all_rewards = []
mean_rewards = deque([0.0 for i in range(100)], maxlen=100)  # only the latest 100 episode
average_q = []
save_path = "models/spaceinvaders_dqn"
outfile_path = "models/output_spaceinvaders_dqn.txt"
total_episodes = 100000
best_reward_decay = 0.9999
save_model_steps = 1000000
warmup_steps = 50000
lr_decay_epoch = 10000

# dev or server mode
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))
device_name = torch.cuda.get_device_name(0)
print("Device name: " + device_name)
if device_name == 'GeForce GTX 1070':
    print("Running in DEV_MODE!")
    MEMORY_SIZE = 500
    warmup_steps = 0
else:
    print("Running in SERVER_MODE!")

STATE_SIZE = env.observation_space
print('State size:', STATE_SIZE)
ACTION_SIZE = env.action_space.n
print('Action size:', ACTION_SIZE)


def np_random_crop(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def preprocess_frame(frame):
    # plt.imshow(frame), plt.show()
    frame = rgb2gray(frame)
    # plt.imshow(frame), plt.show()

    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    frame = frame[8:-13, 15:-15]
    # plt.imshow(frame), plt.show()

    # Normalize Pixel Values
    frame = frame / 255.0

    frame = transform.resize(frame, [84, 84])
    # random crop
    # frame = transform.resize(frame, [110, 84])
    # frame = np_random_crop(frame, 84, 84)

    # plt.imshow(frame), plt.show()

    return frame  # 84x84x1 frame


def stack_frames(stack_size, stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

        # new episode, copy the same frame 4x
        for i in range(stack_size):
            stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


def test_process(agent):

    total_steps = 0
    state = env.reset()
    episode_reward = 0

    # Clear stacked_frames
    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
    state, stacked_frames = stack_frames(stack_size, stacked_frames, state, True)

    while True:
        env.render()
        time.sleep(0.01)

        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        # print("instant reward: ", reward)

        if done:
            print("total reward: ", episode_reward)
            break

        next_state, stacked_frames = stack_frames(stack_size, stacked_frames, next_state, False)
        state = next_state
        total_steps += 1


def test():
    env.frameskip = 3

    dqn_agent = DDQN(
        state_size=stack_size, action_size=ACTION_SIZE, eval=True,
        use_cuda=USE_CUDA, epsilon=0.0, ddqn=True,
    )

    dqn_agent.load(save_path+"_latest.h5")
    test_process(dqn_agent)


def train_process(agent, resume_epi, resume_steps):

    total_steps = resume_steps
    steps = []
    episodes = []
    best_reward = -1
    best_episode = -1
    outfile = open(outfile_path, 'w')

    for i_episode in range(resume_epi, total_episodes):
        state = env.reset()
        episode_reward = 0

        # Clear stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
        state, stacked_frames = stack_frames(stack_size, stacked_frames, state, True)

        while True:
            # env.render()

            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward  # record actual reward
            reward = np.clip(reward, -1., 1.)
            # print("instant reward: ", reward)

            if done:
                all_rewards.append(episode_reward)
                mean_reward = np.mean(all_rewards)
                mean_rewards.append(mean_reward)
                ave_q = agent.get_ave_q()
                average_q.append(ave_q)
                next_state = np.zeros((84, 84), dtype=np.int)
                if episode_reward > best_reward * (best_reward_decay ** (i_episode - best_episode)):
                    best_reward = episode_reward
                    best_episode = i_episode
                    agent.save(save_path+"_best.h5", i_episode, total_steps)
                agent.save(save_path + "_latest.h5", i_episode, total_steps)  # save the latest model

                # print output
                outfile.write('Episode {} done: rewards={}, best_reward={}, mean_reward={},'
                              ' ave_q={}, total_steps={}, lr={}, epsilon={}\n'.
                              format(i_episode, episode_reward, best_reward, mean_reward,
                                     ave_q, total_steps, agent.get_lr(), agent.get_epsilon()))
                outfile.flush()

            if total_steps % save_model_steps == 0 and total_steps > 0:
                agent.save(save_path + "_" + str(total_steps) + "iter.h5", i_episode, total_steps)

            next_state, stacked_frames = stack_frames(stack_size, stacked_frames, next_state, False)
            agent.store_transition(state, action, reward, next_state)

            if total_steps > warmup_steps and total_steps % skip_learning == 0:
                agent.learn(i_episode)

            if done:
                steps.append(total_steps)
                episodes.append(i_episode)
                agent.episode_done()
                break

            state = next_state
            total_steps += 1
    return np.vstack((episodes, steps))


def train():

    dqn_agent = DDQN(
        state_size=stack_size, action_size=ACTION_SIZE, buffer_size=MEMORY_SIZE,
        use_cuda=USE_CUDA, epsilon=1.0, epsilon_decay=0.9999997, lr=.00025, batch_size=32,
        gamma=0.99, lr_decay_epoch=lr_decay_epoch, ddqn=True,
    )

    resume_epi = 0
    resume_steps = 0
    # RESUME_PATH = "pretrained_models/spaceinvaders_dqn_14812104.h5"
    RESUME_PATH = "None"
    if os.path.exists(RESUME_PATH):
        resume_epi, resume_steps = dqn_agent.load(RESUME_PATH)

    his_dqn = train_process(dqn_agent, resume_epi, resume_steps)

    # plot
    plt.plot(his_dqn[0, :], all_rewards, c='b', label='DQN')
    plt.legend(loc='best')
    plt.ylabel('total reward')
    plt.xlabel('episode')
    plt.grid()
    plt.show()

    plt.plot(his_dqn[0, :], mean_rewards, c='b', label='DQN')
    plt.legend(loc='best')
    plt.ylabel('mean reward')
    plt.xlabel('episode')
    plt.grid()
    plt.show()

    plt.plot(his_dqn[0, :], average_q, c='b', label='DQN')
    plt.legend(loc='best')
    plt.ylabel('average q')
    plt.xlabel('episode')
    plt.grid()
    plt.show()


def main():
    # train()
    test()


if __name__ == '__main__':
    main()
