import gym
# from RL_algo.dqn_pytorch import DQN
# from RL_algo.dqn_memory import DQN
# from RL_algo.ddqn_pytorch import DDQN
from RL_algo.ddqn_priority_pytorch import DDQN_Prio
import matplotlib.pyplot as plt
import numpy as np


# env init
env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(21)
MEMORY_SIZE = 2000
all_rewards = []
mean_rewards = []

STATE_SIZE = env.observation_space.shape[0]
print('State size:', STATE_SIZE)
ACTION_SIZE = env.action_space.n
print('Action size:', ACTION_SIZE)


def train(agent):

    total_steps = 0
    steps = []
    episodes = []
    total_episodes = 100000
    best_reward = -1

    for i_episode in range(total_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            if best_reward > 50:
                env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            # modify the reward
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            episode_reward += reward
            reward = np.clip(reward, -1., 1.)
            agent.store_transition(state, action, reward, next_state)
            agent.learn()

            if done:
                all_rewards.append(episode_reward)
                mean_reward = np.mean(all_rewards)
                mean_rewards.append(mean_reward)
                if episode_reward > best_reward:
                    best_reward = episode_reward

                print('Episode {} done: rewards={}, best_reward={}, mean_reward={},'
                      ' total_steps={}, epsilon={}'.
                      format(i_episode, episode_reward, best_reward, mean_reward,
                             total_steps, agent.exploration))
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            state = next_state
            total_steps += 1
    return np.vstack((episodes, steps))


def main():

    dqn_agent = DDQN_Prio(
        state_size=STATE_SIZE, action_size=ACTION_SIZE, buffer_size=MEMORY_SIZE, exploration=0.2,
        lr=0.01, ddqn=True, prioritized=True,
    )

    his_dqn = train(dqn_agent)

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


if __name__ == '__main__':
    main()
