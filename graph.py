import numpy as np
import os.path
import math

from random import randint


class Graph:

    def __init__(self):
        self.state_size = 3
        self.action_size = 2
        self.time = 0
        self.interval = 20
        self.score = 0
        self.price = 0
        self.owned = False
        self.offset = randint(0, self.interval)

    def get_state(self):
        if self.owned:
            state = [self.price]
        else:
            state = [0.0]
        for i in range(self.state_size - len(state)):
            state.append(self.spot_at(self.time - i))
        return state

    def get_score(self):
        return self.score

    def move(self, x):

        if x == 0 and not self.owned:
            self.owned = True
            self.price = self.spot()

        if x == 1 and self.owned:
            self.owned = False
            self.score += self.spot() - self.price

        self.time += 1

    def spot(self):
        return math.fabs(math.sin(math.pi * 2 * (self.time + self.offset) / self.interval))

    def spot_at(self, x):
        return math.fabs(math.sin(math.pi * 2 * (x + self.offset) / self.interval))

    def is_finished(self):
        return self.time >= self.interval

    def moves(self):
        return [0, 1]
