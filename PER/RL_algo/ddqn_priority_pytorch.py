"""
Bill Wang
Mar-2019
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


class Net(nn.Module):
    def __init__(self, in_c, out_c):
        super(Net, self).__init__()
        c = in_c
        layers = []
        for h in [128, 64]:
            layers.append(nn.Linear(c, h))
            # layers[-1].weight.data.normal_(0, 0.1)  # initialization
            # nn.init.xavier_uniform_(layers[-1].weight, gain=1)
            layers.append(nn.ReLU(inplace=True))
            c = h
        layers.append(nn.Linear(c, out_c))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        actions_value = self.layers(x)
        return actions_value


class DDQN_Prio(object):
    def __init__(self, state_size, action_size, buffer_size=2000,
                 exploration=0.5, lr=0.001, batch_size=32, gamma=0.9,
                 ddqn=True, prioritized=True,):
        self.eval_net, self.target_net = Net(state_size, action_size),\
                                         Net(state_size, action_size)
        self.exploration = exploration
        self.action_size = action_size
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self._target_replace_iter = 100
        self.gamma = gamma  # reward decay
        self.ddqn = ddqn
        self.prioritized = prioritized

        self.learn_step_counter = 0
        self.memory_counter = 0

        # initialize memory
        if prioritized:
            self.memory = Memory(capacity=buffer_size)
        else:
            self.memory = np.zeros((buffer_size, state_size * 2 + 2))

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.cost_his = []

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
        if self.prioritized:
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)  # have high priority for newly arrived transition
        else:
            transition = np.hstack((s, [a, r], s_))
            # replace the old memory with new memory
            index = self.memory_counter % self.buffer_size
            self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter < self.buffer_size:
            return

        # target parameter update
        if self.learn_step_counter % self._target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        if self.prioritized:
            tree_idx, b_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.buffer_size, self.batch_size)
            b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :self.state_size])
        b_a = torch.LongTensor(b_memory[:, self.state_size:self.state_size+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_size+1:self.state_size+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_size:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_eval4next = self.eval_net(b_s_).detach()  # next state
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        ### DOUBLE DQN Logic
        # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
        # Use TargetNetwork to calculate the Q_val of Q(s',a')
        if self.ddqn:
            max_act4next = q_eval4next.argmax(1)  # the action that brings the highest value is evaluated by q_eval
            # Double DQN, select q_next depending on above actions
            selected_q_next = torch.stack([q_next[i][max_act4next[i]] for i in batch_index]).unsqueeze(1)
        else:
            selected_q_next = q_next.max(1)[0].view(self.batch_size, 1)

        q_target = b_r + self.gamma * selected_q_next   # shape (batch, 1)

        if self.prioritized:
            abs_errors = torch.sum(torch.abs(q_target - q_eval), dim=1)
            loss = torch.mean(torch.FloatTensor(ISWeights) * ((q_eval - q_target) ** 2))
            self.memory.batch_update(tree_idx, abs_errors.data)  # update priority
        else:
            loss = self.loss_func(q_eval, q_target)

        self.cost_his.append(loss)

        # back_propogation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.exploration > 0.01:
            self.exploration *= 0.99999

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

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


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree

    epsilon = 0.01  # small amount to avoid zero priority
    # tradeoff between taking only exp with high priority and sampling randomly
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """
    def store(self, transition):
        # Find the max priority
        max_p = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

        # print("tree.total_p == ", self.tree.total_p)

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self, n):
        # print("enter sample")

        # Create a sample array that will contains the minibatch
        b_idx, b_memory, ISWeights = \
            np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        pri_seg = self.tree.total_p / n       # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        # Should be calculating the max_weight, min instead
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = pri_seg * i, pri_seg * (i + 1)
            # print("a == ", a, "b == ", b, "n == ", n, "i == ", i)
            v = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            idx, p, data = self.tree.get_leaf(v)

            # P(j)
            prob = p / self.tree.total_p

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    """
    Update the priorities on the tree
    """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity  # for all priority values

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity

        """
        tree:

            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    """
    Here we add our priority score in the sumTree leaf and add the experience in data
    """
    def add(self, p, data):
        # Look at what index we want to put the experience
        tree_idx = self.data_pointer + self.capacity - 1

        """ tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """

        # update data_frame
        self.data[self.data_pointer] = data

        # update tree_frame
        self.update(tree_idx, p)

        # Add 1 to data_pointer
        self.data_pointer += 1

        # If we're above the capacity, you go back to first index (we overwrite)
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree
    """
    def update(self, tree_idx, p):
        # Change = new priority score - former priority score
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        # then propagate the change through tree
        while tree_idx != 0 and change != 0:    # this method is faster than the recursive loop
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """

            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root
