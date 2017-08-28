import numpy as np
import os.path
import math


class Graph:

    def __init__(self):
        self.state_size = 11
        self.action_size = 3
        self.time = 0
        self.interval = 20
        self.score = 0
        self.price = 0
        self.owned = False
        self.illegal_move = False

    def get_state(self):
        if  self.owned:
            state = [1.0, self.price, self.interval]
        else:
            state = [0.0, 0.0, self.interval]
        for i in range(self.state_size - 3):
            state.append(self.spot_at(self.time - i))
        return state

    def get_score(self):
        return self.score

    def move(self, x):

        if x == 0:
            if not self.owned:
                self.owned = True
                self.price = self.spot()
            else:
                self.illegal_move = True
                self.score = -1

        if x == 1:
            if self.owned:
                self.owned = False
                self.score += self.spot() - (self.price + 0.01)
            else:
                self.illegal_move = True
                self.score = -1

        self.time += 1

    def spot(self):
        return math.fabs(math.sin(math.pi * 2 * self.time / self.interval))

    def spot_at(self, x):
        return math.fabs(math.sin(math.pi * 2 * x / self.interval))

    def is_finished(self):
        return self.illegal_move or self.time >= self.interval

    def moves(self):
        if self.owned:
            return [1, 2]
        else:
            return [0, 2]

    def illegal(self):
        return self.illegal_move