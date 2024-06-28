"""
UW, CSEP 573, Win19
"""
from pomdp import POMDP
from offlineSolver import OfflineSolver
import numpy as np

class QMDP(OfflineSolver):
    def __init__(self, pomdp, precision = .000000001):
        super(QMDP, self).__init__(pomdp, precision)
        # offline calculation of the policy following MDP value iteration:
        max_iter = 1000
        epsilon = 1e-6
        # for i in range(max_iter):
        #     pomdp.values = np.copy(new_values) # make sure values update (useless in round 1 ?)
        #     for start_state in range(len(pomdp.states)):
        #         poss_values = np.empty(len(pomdp.actions)) # for each state, there is a matrix of new values for each action
        #         for action in range(len(pomdp.actions)):
        #             # okay in the tigers example, the start state is equivalent to reward, but that's not always the case so be ware ?
        #             poss_values[action] = np.dot(pomdp.R[action][start_state][:, 0], pomdp.T[action][start_state]) + pomdp.discount * np.dot(pomdp.T[action][start_state], pomdp.values)
        #         new_values[start_state] = np.max(poss_values, axis=0)
        #     if np.all(np.abs(new_values - pomdp.values) < epsilon): # if reach convergence threshold, end early
        #         break

        # try a slightly more optimized version? we only store values for states not state/action pairs so it'll be slightly different?.

        # new_values = np.zeros(len(pomdp.states))
        # what if i tried to do it like this HAHAH nope i messed smthing up hehe
        new_values = np.zeros(len(pomdp.states))
        for i in range(max_iter):
            pomdp.values = np.copy(new_values)
            new_values = np.max((np.einsum('ase, ase ->as', pomdp.R[:, :, :, 0], pomdp.T) + pomdp.discount * np.dot(pomdp.T, pomdp.values)), axis=0)
            if np.all(np.abs(new_values - pomdp.values) < epsilon):
                break
        self.pomdp = pomdp
        print('mdp solved')
    
    def chooseAction(self, cur_belief):
        pomdp = self.pomdp

        # original slow code DX
        # q_value = np.empty((len(pomdp.actions), len(pomdp.states)))
        # for action in range(len(pomdp.actions)):
        #     for start_state in range(len(pomdp.states)):
        #         q_value[action][start_state] = np.dot(pomdp.R[action][start_state][:, 0], pomdp.T[action][start_state]) + pomdp.discount * np.dot(pomdp.T[action][start_state], pomdp.values)
        
        # just a thought... im bad at linalg so i dunno if this will work but what if i 
        q_value = (np.einsum('ase, ase ->as', pomdp.R[:, :, :, 0], pomdp.T) + pomdp.discount * np.dot(pomdp.T, pomdp.values))
        # ok still a little slow but better maybe?

        cur_belief = cur_belief.reshape(-1) 
        q_value = np.dot(q_value, cur_belief) # theres def a way to do this easier
        action = np.argmax(q_value, axis=0)
        return action
    
    def getValue(self, belief):
        pomdp = self.pomdp

        q_value = (np.einsum('ase, ase ->as', pomdp.R[:, :, :, 0], pomdp.T) + pomdp.discount * np.dot(pomdp.T, pomdp.values))
        q_value = np.dot(q_value, belief)
        value = np.max(q_value)
        return value

    

class MinMDP(OfflineSolver):
    
    def __init__(self, pomdp, precision = .001):
        super(MinMDP, self).__init__(pomdp, precision)
        # is the only difference the policy computation? using min instead of max
        max_iter = 1000
        epsilon = 1e-6
        new_values = np.zeros(len(pomdp.states))
        for i in range(max_iter):
            pomdp.values = np.copy(new_values)
            new_values = np.min((np.einsum('ase, ase ->as', pomdp.R[:, :, :, 0], pomdp.T) + pomdp.discount * np.dot(pomdp.T, pomdp.values)), axis=0)
            if np.all(np.abs(new_values - pomdp.values) < epsilon):
                break
        self.pomdp = pomdp
    
    def getValue(self, cur_belief):
        # is this literally the same thing?
        pomdp = self.pomdp
        q_value = (np.einsum('ase, ase ->as', pomdp.R[:, :, :, 0], pomdp.T) + pomdp.discount * np.dot(pomdp.T, pomdp.values))
        q_value = np.dot(q_value, cur_belief)
        value = np.max(q_value)
        return value
    

    def chooseAction(self, cur_belief):
        pomdp = self.pomdp
        q_value = (np.einsum('ase, ase ->as', pomdp.R[:, :, :, 0], pomdp.T) + pomdp.discount * np.dot(pomdp.T, pomdp.values))
        cur_belief = cur_belief.reshape(-1) 
        q_value = np.dot(q_value, cur_belief) 
        action = np.argmax(q_value, axis=0)
        return action
    