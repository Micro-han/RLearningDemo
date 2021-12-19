import numpy as np
import gym
import time


def get_action(weights, status):
    weight = np.dot(weights[:4], status) + weights[4]
    if weight >= 0:
        return 1
    else:
        return 0


def get_sum_reward_by_weights(env, weights):
    status = env.reset()
    sum_reward = 0
    for i in range(1000):
        action = get_action(weights, status)
        status, reward, done, info = env.step(action)
        sum_reward += reward
        if done:
            break
    return sum_reward


def get_weights_by_random_guess():
    return np.random.rand(5)


def get_weights_by_hilld_climbing(best_weights):
    return best_weights + np.random.normal(0, 0.1, 5)


def get_best_result(algo="random_guess"):
    env = gym.make("CartPole-v0")
    np.random.seed(10)
    best_reward = 0
    best_weights = np.random.rand(5)
    for i in range(10000):
        cur_weights = None
        if algo == "hill_climbing":
            cur_weights = get_weights_by_hilld_climbing(best_weights)
        else:
            cur_weights = get_weights_by_random_guess()

        cur_sum_reward = get_sum_reward_by_weights(env, cur_weights)
        if cur_sum_reward > best_reward:
            best_reward = cur_sum_reward
            best_weights = cur_weights
        if best_reward >= 200:
            break

    print(iter, best_reward, best_weights)
    return best_reward, best_weights


print(get_best_result("hill_climbing"))