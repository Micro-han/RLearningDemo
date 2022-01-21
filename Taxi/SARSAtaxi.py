import random
import gym


env = gym.make('Taxi-v3')

alpha = 0.85
gamma = 0.90
epsilon = 0.8

q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s, a)] = 0.0


def epsilon_greedy(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: q[(state, x)])


n_iterator = 8000
for i in range(n_iterator):
    r = 0
    state = env.reset()
    action = epsilon_greedy(state)
    while True:
        next_state, reward, done, info = env.step(action)
        next_action = epsilon_greedy(next_state)
        q[(state, action)] += alpha * (reward + gamma * q[(next_state, next_action)] - q[(state, action)])
        action = next_action
        state = next_state
        r += reward
        if done:
            break

    print("total reward:", r)

env.close()