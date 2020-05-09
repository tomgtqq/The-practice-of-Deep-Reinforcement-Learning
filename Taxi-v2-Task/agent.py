import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.27, gamma=0.76):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma        
        
    def epsilon_greedy_probs(self, state, i_episode = 1, eps = None):
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - epsilon + (epsilon / self.nA)
        
        return policy_s
 

    def select_action(self, state, policy_s):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment
        - policy_s: the policy with Q-table
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(np.arange(self.nA), p=policy_s)
            
   
    def step(self, state, action, reward, next_state, done, policy_s):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
#         Q[state][action] = update_Q(Q[state][action], np.dot(Q[next_state], policy_s), reward, alpha, gamma)
#         Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa)
#  expected sarsa
        self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward + (self.gamma * np.dot(self.Q[next_state], policy_s)) - self.Q[state][action]))
#  Q-learning
#         self.Q[state][action] = (1 - self.alpha)*self.Q[state][action] + self.alpha*(reward + np.max(self.Q[next_state]))
#         self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward + (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action]))