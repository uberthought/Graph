import tensorflow as tf
import numpy as np
import os.path
import math

from graph import Graph
from network import DNN

graph = Graph()
dnn = DNN(graph.state_size, graph.action_size)

sum = 0
total = 100

for i in range(1, total):
    while not graph.is_finished():

        state = graph.get_state()
        actions = dnn.run([state])
        action = np.argmax(actions)
        graph.move(action)

        print()
        print('state ', state)
        print('actions ', actions, ' ', action)
        # print('action ', action, ' score ', graph.get_score())
        # print('spot ', graph.spot(), ' action ', action, ' score ', graph.get_score())

    sum += graph.get_score()
    # print('Offset: ', graph.offset)
    # print('Spot: ', graph.spot())
    print('Score: ', graph.get_score())

    graph = Graph()

print('Average: ', sum / total)
