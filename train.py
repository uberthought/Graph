import tensorflow as tf
import numpy as np
import os.path
import math
import pickle

from graph import Graph
from network import DNN
from random import randint

graph = Graph()
dnn = DNN(graph.state_size, graph.action_size)

experiences = []
old_experiences = []
if os.path.exists('old_experiences.p'):
    old_experiences = pickle.load(open("old_experiences.p", "rb"))


def train(dnn, experiences):
    X = np.array([], dtype=np.float).reshape(0, graph.state_size)
    Y = np.array([], dtype=np.float).reshape(0, graph.action_size)

    for experience in experiences:
        state0 = experience['state0']
        action = experience['action']
        state1 = experience['state1']
        score = experience['score']
        terminal = experience['terminal']

        actions1 = dnn.run([state0])

        if terminal:
            actions1[0][action] = score
        else:
            actions2 = dnn.run([state1])
            discount_factor = 1
            actions1[0][action] = score + discount_factor * np.max(actions2)

        X = np.concatenate((X, np.reshape(state0, (1, graph.state_size))), axis=0)
        Y = np.concatenate((Y, actions1), axis=0)

    return dnn.train(X, Y)


def add_experience(experience):

    index = len(old_experiences) - 1
    for existing in reversed(old_experiences):
        if existing['state0'] == experience['state0']:
            old_experiences.pop(index)
        index -= 1
    old_experiences.append(experience)

    experiences.append(experience)


print('old_experiences ', len(old_experiences))

rounds = 0

# For life or until learning is stopped...
for i in range(10000000):

    state0 = graph.get_state()

    # player move
    state0 = graph.get_state()
    actions = dnn.run([state0])
    action = np.argmax(actions)

    if randint(0, 10) == 0:
        moves = graph.moves()
        action = np.random.choice(moves, 1)

    # Take the action (aa) and observe the the outcome state (s′s′) and reward (rr).
    graph.move(action)

    state1 = graph.get_state()
    terminal = graph.is_finished()
    score = graph.get_score()

    experience = {'state0': state0, 'action': action, 'state1': state1, 'score': score, 'terminal': terminal}
    add_experience(experience)

    if terminal:

        # switch to next graph

        rounds += 1
        graph = Graph()

        # add old experiences

        if len(old_experiences) >= 0:
            random_old_experiences = np.random.choice(old_experiences, len(experiences) * 4).tolist()
            experiences = experiences + random_old_experiences

        # train
        loss = train(dnn, experiences)

        print('rounds ', rounds, ' loss ', loss, ' score ', score)

        experiences = []

        pickle.dump(old_experiences, open("old_experiences.p", "wb"))
        dnn.save()
