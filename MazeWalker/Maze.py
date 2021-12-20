import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

ACTIONS = ['←', '→', '↑', '↓']
threshold = 0.05


def move_one_step(current, action, shape):
    current = np.copy(current)
    if action == '↑':
        current[0] = max(current[0] - 1, 0)
    if action == '↓':
        current[0] = min(current[0] + 1, shape[0] - 1)
    if action == '←':
        current[1] = max(current[1] - 1, 0)
    if action == '→':
        current[1] = min(current[1] + 1, shape[1] - 1)
    return current


def _visualize_one_matrix(matrix, action_matrix, title='', formatter='{:0.01f}', colormap='hot'):
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.imshow(matrix, cmap=colormap, )
    plt.title(title)
    plt.colorbar()

    display_matrix = np.array([formatter.format(val) + '\n' + action
                              for val, action in zip(matrix.ravel(), action_matrix.ravel())]).reshape(matrix.shape)
    for (i, j), z in np.ndenumerate(display_matrix):
        ax.text(j, i, z, ha='center', va='center', size=15)

    plt.savefig(title+'.jpg', dpi=300)


class MazeAgent:
    def __init__(self, rewards, start=None, end=None, gamma=0.95, barrierThreshold=-50):
        self._maze = rewards
        self._shape = np.array(rewards.shape)
        self._start = start if start is not None else np.array((0, 0))
        self._end = end if end is not None else self._shape - np.array((1, 1))
        self.value = np.zeros(rewards.shape)
        self.action_value = np.zeros(list(rewards.shape) + [4])
        self.gamma = gamma
        self.barrierThreshold = barrierThreshold

    def _my_init(self, newStart=None):
        self._current = np.copy(self._start) if newStart is None else newStart
        self._policy = np.array(random.choices(ACTIONS, k=self._shape[0] * self._shape[1])).reshape(self._shape)
        self._initialPolicy = np.copy(self._policy)
        self._policyZero = np.copy(self._policy)

    def _walk_through_epsilon(self, maxIter=50000, policy=None):
        count = 0
        policy = np.copy(self._policy) if policy is None else policy

        while not np.array_equal(self._current, self._end) and count < maxIter:
            self._prevCurrent = np.copy(self._current)
            epsilon = np.random.uniform()
            if epsilon < threshold:
                policy[self._current[0], self._current[1]] = random.choice(ACTIONS)
            self._current = move_one_step(self._current, policy[self._current[0], self._current[1]], self._shape)
            if self._maze[self._current[0], self._current[1]] == -np.infty:
                self._current = np.copy(self._prevCurrent)
            count = count + 1
        return policy

    def _policy_evaluation(self, iterations=100, policy=None):
        barrier_threshold = self.barrierThreshold
        policy = self._policy if policy is None else policy
        for i in range(iterations):
            value_current = np.copy(self.value)
            for row in range(self._shape[0]):
                for col in range(self._shape[1]):
                    if np.array_equal([row, col], self._end) or self._maze[row, col] < barrier_threshold:
                        self.value[row, col] = self._maze[row, col]
                        continue
                    action = policy[row, col]
                    next_state = move_one_step(np.array([row, col]), action, self._shape)
                    if self._maze[next_state[0], next_state[1]] < barrier_threshold:
                        next_state = np.array([row, col])
                    self.value[row, col] = self._maze[next_state[0], next_state[1]] + self.gamma * value_current[next_state[0], next_state[1]]
        return self.value

    def _policy_improvement_from_value(self):
        for row in range(self._shape[0]):
            for col in range(self._shape[1]):
                candidate_rewards = []
                for action in ACTIONS:
                    if np.array_equal([row, col], self._end) or self._maze[row, col] < self.barrierThreshold:
                        next_state = np.array([row, col])
                    else:
                        next_state = move_one_step((row, col), action, self._shape)

                    reward = self._maze[next_state[0], next_state[1]] + self.gamma * self.value[
                        next_state[0], next_state[1]]
                    candidate_rewards.append(reward)

                re_index = np.argmax(candidate_rewards)
                self._policy[row, col] = ACTIONS[re_index]

    def run_policy_iteration_from_value(self, episodes=100, evaluation_iters=50, printMaze=False):
        for it in range(episodes):
            self._policy_evaluation(iterations=evaluation_iters)
            self._policy_improvement_from_value()
            if printMaze:
                _visualize_one_matrix(self.value, action_matrix=self._policy, colormap='PiYG', title='Value function and policy after learning')

    def _off_policy_one_episode(self, alpha=0.25, epsilon=0.95, epsilonNextActSARSA=0.95, method='qLearning'):
        self._current = np.copy(self._start)
        while not np.array_equal(self._current, self._end):
            current = self._current
            action = ACTIONS[self.action_value[self._current[0], self._current[1]].argmax()]
            rand = np.random.uniform()
            if rand > epsilon:
                action = random.choice(pd.Index(ACTIONS).difference([action]))
            next_state = move_one_step(self._current, action, self._shape)
            action_index = ACTIONS.index(action)

            if self._maze[next_state[0], next_state[1]] < self.barrierThreshold:
                next_state = current
            reward = self._maze[next_state[0], next_state[1]]
            if method == 'qLearning':
                new_val = self.action_value[next_state[0], next_state[1]].max()
            else:
                next_action_index = self.action_value[next_state[0], next_state[1]].argmax()
                rand = np.random.uniform()
                if rand > epsilonNextActSARSA:
                    next_action_index = random.choice(pd.Index(range(len(ACTIONS))).difference([next_action_index]))

                new_val = self.action_value[next_state[0], next_state[1], next_action_index]
            self.action_value[current[0], current[1], action_index] = self.action_value[current[0], current[1], action_index] + alpha * (reward + self.gamma * new_val - self.action_value[current[0], current[1], action_index])
            self._current = next_state
        self._policy = np.vectorize(lambda x: ACTIONS[x])(self.action_value.argmax(axis=2))
        self.value = self.action_value.max(axis=2)

    def run_off_policy_learning(self, episodes=100, alpha=0.25, epsilon=0.95, epsilonNextActSARSA=0.95, method='qLearning'):
        self.action_value[self._end[0], self._end[1], :] = self._maze[self._end[0], self._end[1]]
        for ep in range(episodes):
            self._off_policy_one_episode(alpha, epsilon=epsilon, epsilonNextActSARSA=epsilonNextActSARSA, method=method)

    def visualize(self):
        _visualize_one_matrix(self._maze, action_matrix=self._policyZero, title='Maze and initial policy')
        _visualize_one_matrix(self.value, action_matrix=self._policy, colormap='PiYG', title='Value function and policy after learning')


maze0 = np.array([[-1, -1, -1, -1, -1],
                  [-1, -np.infty, -np.infty, -np.infty, -1],
                  [-1, -1, -1, -1, -1],
                  [-10, -np.infty, -1, -np.infty, -np.infty],
                  [-1, -1, -1, -1, 10]])

print(maze0)
# dp
# agent0 = MazeAgent(maze0, end=(4, 4))
# agent0._my_init()
# agent0.run_policy_iteration_from_value(episodes=10)
# agent0.visualize()
# q-learning
agent0 = MazeAgent(maze0, end=(4, 4))
agent0._my_init()
agent0.run_off_policy_learning(episodes=1000, alpha=0.25, epsilon=0.5, method='qLearning')
agent0.visualize()
# SARSA
# agent0 = MazeAgent(maze0, end=(4, 4))
# agent0._my_init()
# agent0.run_off_policy_learning(episodes=1000, alpha=0.25, epsilon=0.5, method='Vive La France')
# agent0.visualize()
