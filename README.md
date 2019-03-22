# reinforcement learning
Compare Sarsa algorithm with Qlearning algorithm.

For detail comparison，go to https://blog.csdn.net/qq_39004117/article/details/81705845 (in Chinese)

The basic environment described as follows:
The agent is in a maze. Given the entrance and exit of the maze, he should find the exit without falling into the snare. If he arrives at the destination, then the game is over. And if he goes into the snare, then he will get -100 reward, then restart from the entrance. For each step, he will get -1 reward.
the map of the game shown as follows:
![game map](https://github.com/nuomizai/Qlearning/blob/master/image/map.jpg)
Run this algorithm, you need to install the following python lib:
- matplotlib
- numpy
