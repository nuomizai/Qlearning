import numpy as np
from environment import CliffEnvironment
import sys
from collections import defaultdict
import matplotlib.pyplot as plt


# if best_action = 0, then the probability of action is:
# A[epsilon/4, 1-3*epsilon/4, epsilon/4, epsilon/4]
def epsilon_greedy_policy(Q, state, nA, epsilon=0.1):
    best_action = np.argmax(Q[state])
    A = np.ones(nA, dtype=np.float32) * epsilon / nA
    A[best_action] += 1 - epsilon
    return A


def plot(x, y):
    size = len(x)
    x = [x[i] for i in range(size) if i % 50 == 0]
    y = [y[i] for i in range(size) if i % 50 == 0]
    plt.plot(x, y, 'ro-')
    plt.ylim(-300, 0)
    plt.show()


def print_policy(Q):
    env = CliffEnvironment()
    result = ""
    for i in range(env.height):
        line = ""
        for j in range(env.width):
            action = np.argmax(Q[(j, i)])  # find the action to max Q function
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


def sara(env, episode_nums, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    env = CliffEnvironment()
    Q = defaultdict(lambda: np.zeros(env.nA))
    rewards = []
    # policy = []
    for episode in range(episode_nums):  # episode_nums: 1000
        # if episode % 50 == 0:
        #     policy.append(np.argmax(Q[tuple((2, 2))]))

        env._reset()
        state, done = env.observation()
        A = epsilon_greedy_policy(Q, state, env.nA)
        probs = A
        action = np.random.choice(np.arange(env.nA), p=probs)  # action probability
        sum_reward = 0.0

        while not done:
            next_state, reward, done = env._step(action)  # exploration

            if done:
                Q[state][action] = Q[state][action] + alpha * (reward + discount_factor * 0.0 - Q[state][action])
                break
            else:
                next_A = epsilon_greedy_policy(Q, next_state, env.nA)  # get action probability distribution for next state
                probs = next_A
                next_action = np.random.choice(np.arange(env.nA),
                                               p=probs)  # get next action, use [next_state][next_action]  to update Q[state][action]
                Q[state][action] = Q[state][action] + alpha * (
                        reward + discount_factor * Q[next_state][next_action] - Q[state][action])
                state = next_state
                action = next_action
            sum_reward += reward
        rewards.append(sum_reward)

    # plot(range(1,1+ len(policy)),policy)
    return Q, rewards


env = CliffEnvironment()
episode_nums = 1000
Q, rewards = sara(env, episode_nums)

average_rewards = []

for i in range(10):
    Q, rewards = sara(env, episode_nums)
    if len(average_rewards) == 0:
        average_rewards = np.array(rewards)
    else:
        average_rewards = average_rewards + np.array(rewards)

average_rewards = average_rewards * 1.0 / 10
plt.title('average_rewards_for_Sarsa')
plot(range(episode_nums), average_rewards)

print_policy(Q)

# As is shown from the result，TD（0）is a conservative policy, for it calculates the safest way to the destination
