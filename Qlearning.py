import numpy as np
from collections import defaultdict
from environment import CliffEnvironment
import matplotlib.pyplot as plt


def epsilon_greedy_policy(Q, state, nA, epsilon=0.1):
    action = np.argmax(Q[state])
    A = np.ones(nA, dtype=np.float32) * epsilon / nA
    A[action] += 1 - epsilon
    return A


def greedy_policy(Q, state):
    best_action = np.argmax(Q[state])
    return best_action


def print_policy(Q):
    env = CliffEnvironment()
    result = ""
    for i in range(env.height):
        line = ""
        for j in range(env.width):
            action = np.argmax(Q[(j, i)])  # 找到具有最大Q的action
            if action == 0:
                line += "up  "
            elif action == 1:
                line += "down  "
            elif action == 2:
                line += "left  "
            else:
                line += "right  "
        result = line + "\n" + result
    print(result)


def Qlearning(alpha=0.1, episode_num=1000, discount_factor=1.0):
    env = CliffEnvironment()
    Q = defaultdict(lambda: np.zeros(env.nA))  # lambda的作用是什么
    rewards = []
    for i in range(episode_num):
        env._reset()
        cur_state, done = env.observation()
        sum_reward = 0.0

        while not done:
            prob = epsilon_greedy_policy(Q, cur_state, env.nA)
            action = np.random.choice(np.arange(env.nA), p=prob)  # 每次重新决定当前的action
            next_state, reward, done = env._step(action)  # 使用action找到下一个状态
            if done:
                Q[cur_state][action] = Q[cur_state][action] + alpha * (
                        reward + discount_factor * 0.0 - Q[cur_state][action])
                break
            else:
                next_action = greedy_policy(Q,
                                            cur_state)  # 找到下一个action,使用Q[next_state][next_action]的值，但是下一个action不一定是这个
                # next_action = greedy_policy(Q,next_state)
                Q[cur_state][action] = Q[cur_state][action] + alpha * (
                        reward + discount_factor * Q[next_state][next_action] - Q[cur_state][action])
                cur_state = next_state
            sum_reward += reward
        rewards.append(sum_reward)
    return Q, rewards


def plot(x, y):
    size = len(x)
    x = [x[i] for i in range(size) if i % 50 == 0]
    y = [y[i] for i in range(size) if i % 50 == 0]
    plt.plot(x, y, 'ro-')
    plt.ylim(-300, 0)
    plt.show()


Q, rewards = Qlearning()
print_policy(Q)
average_rewards = []
for i in range(10):
    _, rewards = Qlearning()
    if len(average_rewards) == 0:
        average_rewards = np.array(rewards)
    else:
        average_rewards = average_rewards + np.array(rewards)
average_rewards = average_rewards * 1.0 / 10
plot(range(1, 1 + 1000), average_rewards)
