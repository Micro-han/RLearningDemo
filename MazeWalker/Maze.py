import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

ACTIONS = ['L', 'R', 'U', 'D']


def moveOneStep(current, action, shape):
    current = np.copy(current)
    if action == 'U':
        current[0] = max(current[0] - 1, 0)
    if action == 'D':
        current[0] = min(current[0] + 1, shape[0] - 1)
    if action == 'L':
        current[1] = max(current[1] - 1, 0)
    if action == 'R':
        current[1] = min(current[1] + 1, shape[1] - 1)
    return current


class MazeAgent:
    def __init__(self, maze, start=None, end=None, gamma=0.95, barrierThreshold=-50):
        self._maze = maze
        self._shape = np.array(maze.shape)
        self._start = start if start is not None else np.array((0, 0))
        self._end = end if end is not None else self._shape - np.array((1, 1))
        self._gamma = gamma
        self._value = np.zeros(maze.shape)
        self.actionValue = np.zeros(list(maze.shape) + [4])
        self.barrierThreshold = barrierThreshold

    def _initialize(self, newStart=None):
        self._current = np.copy(self._shape) if newStart is None else newStart
        self._policy = np.array(random.choices(ACTIONS, k=self._shape[0]*self._shape[1])).reshape(self._shape)
        self._initializePolicy = np.copy(self._policy)
        self._policyZero = np.copy(self._policy)

    def _walkThroughEpsilon(self, maxIter=50000, policy=None):
        count = 0
        policy = np.copy(self._policy) if policy is None else policy
