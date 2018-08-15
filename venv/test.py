import numpy as np
from collections import defaultdict

Q = defaultdict(lambda: np.zeros(4))
Q[tuple((0, 0))][0] = 1
Q[tuple((0, 0))][1] = 2
Q[tuple((0, 0))][2] = 3
Q[tuple((0, 0))][3] = 5
state = tuple((0, 0))
action = np.argmax(Q[state])
print(action)
print(Q[state])
