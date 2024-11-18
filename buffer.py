import random
from typing import List

import numpy as np
from my_types import TRANSITION_TYPE


class Buffer:

    buffer: List[TRANSITION_TYPE]
    N: int

    def __init__(self, N: int):
        self.N = N
        self.index_to_write = 0

        self.buffer = []

    def append(self, e: TRANSITION_TYPE) -> None:

        if len(self.buffer) < self.N:
            self.buffer.append(e)
        else:
            self.buffer[self.index_to_write] = e

        self.index_to_write = (self.index_to_write + 1) % self.N

        assert 0 <= self.index_to_write and self.index_to_write < self.N
        assert len(self.buffer) <= self.N

    def random_sample(self, nb: int) -> List[TRANSITION_TYPE]:
        size = min(len(self.buffer), nb)
        return [self.buffer[i] for i in np.random.randint(low=0, high=1, size=size)]
