import random
import gym


env = gym.make('Taxi-v3')

# alpha 学习率 gamma 折扣因数 epsilon 贪心
# 初始化
alpha = 0.4
gamma = 0.999
epsilon = 0.17

q = {}

for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s, a)] = 0


# 更新q表
def update_q_table(pre_state, action, reward, next_state):
    q_a = max([q[(next_state, a)] for a in range(env.action_space.n)])
    q[(pre_state, action)] += alpha * (reward + gamma * q_a - q[(pre_state, a)])


# epsilon 贪心
def epsilon_greedy_policy(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: q[(state, x)])


n_iterator = 80000
for i in range(n_iterator):
    # 初始化环境
    r = 0
    pre_state = env.reset()
    while True:
        # 根据 epsilon贪心选择行为
        # env.render()
        action = epsilon_greedy_policy(pre_state)
        # 与环境交互 更新q表
        next_state, reward, done, info = env.step(action)
        update_q_table(pre_state, action, reward, next_state)
        # 迭代
        pre_state = next_state
        r += reward
        if done:
            break
    # 检查最终效果
    print("total reward:", r)

env.close()