import gym
from RL_algo.dqn_priority_tf import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# env init
env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(21)
MEMORY_SIZE = 10000


def train(agent):

    total_steps = 0
    steps = []
    episodes = []
    total_episodes = 20

    for i_episode in range(total_episodes):
        state = env.reset()
        while True:
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            if done:
                reward = 10

            agent.store_transition(state, action, reward, next_state)
            if total_steps > MEMORY_SIZE:
                agent.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            state = next_state
            total_steps += 1
    return np.vstack((episodes, steps))


def main():

    sess = tf.Session()
    with tf.variable_scope('natural_DQN'):
        dqn_agent = DQNPrioritizedReplay(
            n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.00005, sess=sess, prioritized=False,
        )

    with tf.variable_scope('DQN_with_prioritized_replay'):
        dqn_agent_prio = DQNPrioritizedReplay(
            n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.00005, sess=sess, prioritized=True,
            # output_graph=True,
        )
    sess.run(tf.global_variables_initializer())

    his_dqn = train(dqn_agent)
    his_prio = train(dqn_agent_prio)

    # compare based on first success
    plt.plot(his_dqn[0, :], his_dqn[1, :] - his_dqn[1, 0], c='b', label='natural DQN')
    plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
