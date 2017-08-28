import tensorflow as tf
import numpy as np
import os.path
import math

from graph import Graph
from network import DNN

graph = Graph()
dnn = DNN(graph.state_size, graph.action_size)

while not graph.is_finished():

    state = graph.get_state()
    actions = dnn.run([state])
    action = np.argmax(actions)
    graph.move(action)

    print('actions ', actions)
    print('action ', action, ' score ', graph.get_score())
    # print('spot ', graph.spot(), ' action ', action, ' score ', graph.get_score())


print('Illegal Move: ', graph.illegal())
print('Score: ', graph.get_score())
