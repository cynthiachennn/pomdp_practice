# script to run random tests on stuff ugh

from pomdp import POMDP
from environment import Environment
from onlineSolver import OnlineSolver
from offlineSolver import OfflineSolver
from policyReader import PolicyReader
from aems import AEMS2
from mdpSolver import QMDP, MinMDP
import numpy as np
import sys


# exploratory env script
# args format: python3 toy.py Tiger
model_file = 'examples/env/' + sys.argv[1] + '.pomdp'
pomdp = POMDP(model_file)

# environment internals for reference :D
# print("Reward Matrix\n", pomdp.R)
# print("Transition Matrix\n", pomdp.T)
# print("Observation Matrix\n", pomdp.T)

# lets try setting up the FIRST ROUND of value iteration.

new_values = np.zeros(len(pomdp.states))

def value_iteration(): # basically same as current QMDP, some var names changed?
    pomdp.values = np.copy(new_values) # make sure values update (useless in round 1)
    for start_state in range(len(pomdp.states)):
        poss_values = np.empty(len(pomdp.actions)) # for each state, there is a matrix of new values for each action
        for action in range(len(pomdp.actions)):
            # okay in the tigers example, the start state is equivalent to reward, but that's not always the case so be ware ?
            poss_values[action] = np.dot(pomdp.R[action][start_state][:, 0], pomdp.T[action][start_state]) + pomdp.discount * np.dot(pomdp.T[action][start_state], pomdp.values)
        # print(f'Value iteration for state = {start_state}\n', poss_values)
        new_values[start_state] = np.max(poss_values, axis=0)

# new_values = np.empty((len(pomdp.actions), len(pomdp.states)))

# def value_iteration(): # basically same as current QMDP, some var names changed?
#     pomdp.values = np.copy(new_values) # make sure values update (useless in round 1)
#     for start_state in range(len(pomdp.states)):
#         poss_values = np.empty(len(pomdp.actions)) # for each state, there is a matrix of new values for each action
#         for action in range(len(pomdp.actions)):
#             # okay in the tigers example, the start state is equivalent to reward, but that's not always the case so be ware ?
#             poss_values[action] = np.dot(pomdp.R[action][start_state][:, 0], pomdp.T[action][start_state]) + pomdp.discount * np.dot(pomdp.T[action][start_state], pomdp.values)
#         # print(f'Value iteration for state = {start_state}\n', poss_values)
#         new_values[start_state] = np.max(poss_values, axis=0)


# value_iteration() # seems to work for one round. lets try multiple rounds ?

max_iter = 1000 # wow i looooove customizable/modular code hehehe
epsilon = 1e-6
for i in range(max_iter):
    value_iteration()
    # print(new_values)
    if np.all(np.abs(new_values - pomdp.values) < epsilon): # if reach convergence threshold, end early
        # print("num iter:", i) # i + 1 to account for the setup loop hehehe
        break

# WOOOOAH okay something happened hehehe 

#lets try this with a test belief
belief = [0.2, 0.8] # Value by QMDP: 28.98, action by QMDP: 0

#def getValue(belief):
q_value = np.empty((len(pomdp.actions), len(pomdp.states)))
for action in range(len(pomdp.actions)):
    for start_state in range(len(pomdp.states)):
        # sooo i just uhh copied this from before but tbh i think it works?
        # ok one small mod where i do q_value[action][start_state]. instead of just policy[action]
        # scared of changing it up top woo lets try thoo. i think it's cuz i combined the "policy" array (on a state index level i guess) w the bigger "new_values" array in this case.
        q_value[action][start_state] = np.dot(pomdp.R[action][start_state][:, 0], pomdp.T[action][start_state]) + pomdp.discount * np.dot(pomdp.T[action][start_state], pomdp.values)
        # print("rew half", np.dot(pomdp.R[action][start_state][:, 0], pomdp.T[action][start_state]))
        # print("value half", pomdp.discount * np.dot(pomdp.T[action][start_state], pomdp.values))
        # print(f'Q({start_state}, {action}) = {q_value[action]} = {pomdp.R[action][start_state][:, 0]} * {pomdp.T[action][start_state]} + {pomdp.discount} * {pomdp.T[action][start_state]} * {pomdp.values})')
        # print(q_value[action])
    print("before belief?", q_value)
q_value = np.dot(q_value, belief)
print("after belief.", q_value) #OMG I HOPE THIS WORKS AHHHH
value = np.max(q_value)
action = np.argmax(q_value) # j for shiz a gigs

print(value, action)
    # okay yeah i guess the only difference between MDP/QMDP is that do an EXTRA step of multipling by belief, which skews the value calc a bit?





precision, total_reward, environment, time_step = 0.001, 0, Environment(pomdp), 0 # just setup :D
Max_abs_reward = np.max(np.abs(pomdp.R)) # gurl idk
cur_belief = np.array(pomdp.prior).reshape(1, len(pomdp.prior)) # initialize start state

