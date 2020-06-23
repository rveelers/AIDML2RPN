import random
import numpy as np

from collections import deque


class ReplayBuffer:
    """Constructs a buffer object that stores previous state-action-reward pairs and samples a set of sub-samples. """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, d, s2):
        """ Add an experience to the buffer. S represents the current state, a is an action, r is a reward,
        d is whether it is in the done state, and s2 is next state. """

        experience = (s, a, r, d, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size):
        """ Samples a total of elements equal to batch_size from buffer if buffer contains enough elements.
        Otherwise return all elements. """

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch, a_batch, r_batch, d_batch, s2_batch = list(map(np.array, list(zip(*batch))))

        return s_batch, a_batch, r_batch, d_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
