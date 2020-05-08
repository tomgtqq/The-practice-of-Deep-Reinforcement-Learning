# Temporal-Difference Methods

#Part 0 : Explore CliffWalkingEnv
# importing the necessary packages

import sys
import gym
import numpy as np 
from collections import defaultdict, deque
import matplotlib.pyplot as plt 
%matplotlib inline 

import check_test
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')

print(env.action_space)
print(env.observation_space)

# Part1: TD Control Sarsa

# Input: policy π, positive integer num episodes, small positive fraction α, GLIE {εi}
# Output: value function Q (≈ qπ if num episodes is large enough)
# Initialize Q arbitrarily (e.g., Q(s, a) = 0 for all s ∈ S and a ∈ A(s), and Q(terminal-state, ·) = 0)
# for i ← 1 to num episodes do 
# 	ε ← εi
# 	Observe S0
#     Choose action A0 using policy derived from Q (e.g., ε-greedy) t←0
#     repeat
#        Take action At and observe Rt+1 , St+1
#        Choose action At+1 using policy derived from Q (e.g., ε-greedy)
#        Q(St, At) ← Q(St, At) + α(Rt+1 + γQ(St+1, At+1) − Q(St, At)) 
#        t←t+1
#     until St is terminal; 
# end
# return Q


# Q(St, At) ← Q(St, At) + α(Rt+1 + γQ(St+1, At+1) − Q(St, At))
def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
	""" updates the action-value function estimate using the most recent time step """
	return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

# Choose action A0 using policy derived from Q (e.g., ε-greedy)
# Choose action At+1 using policy derived from Q (e.g., ε-greedy)
def epsilon_greedy_probs(env, Q_s, i_episode, eps=None):
	epsilon = 1.0 / i_episode
	if eps is not None:
		epsilon = eps 

	policy_s = np.ones(env.nA) * epsilon / env.nA 
	policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / env.nA)
    return  policy_s

 def sarsa(env, num_episodes, alpha, gamma=1.0):
 	Q = defaultdict(lambda: np.zeros(env.nA))
 	plot_every = 100
 	tmp_scores = deque(maxlen=num_episodes)
 	# loop over episode
 	for i_episode in range(1, num_episodes+1):
		 if i_episode % 100 == 0:
	        print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
	        sys.stdout.flush()   
		 # initialize score
		 score = 0
		 # begin a episode, observe S0
		 state = env.reset()
		 # get epsilon-greedy aciton probabilities
		 policy_s = epsilon_greedy_probs(env, Q[state], i_episode)
		 # pick action A
		 action = np.random.choice(np.arange(env.nA), p=policy_s)
		 # limit number of time steps per episode
		 for t_step in np.arange(300):
		 		# take action A, observe R, S'
		 		next_state, reward, done, info = env.step(action)
		 		# add reward to score
		 		score += reward
		 		if not done:
		 			# get epsilon-greedy action probabilities
		 			policy_s = epsilon_greedy_probs(env, Q[next_state], i_episode)
		 			# pick next action A'
		 			next_action = np.random.choice(np.arange(env.nA), p=policy_s)
		 			# update TD estimate of Q
		 			Q[state][action] = update_Q(Q[state][action], Q[next_state][next_action], reward, alpha, gamma)

		 			# S <- S'
		 			state = next_state
		 			# A <- A'
		 			action = next_action 

		 		if done:
		 			# update TD estimate of Q
		 			Q[state][action] = update_Q(Q[state][action], 0, reward, alpha, gamma)
		 			# append score
		 			tmp_scores.append(score)
		 			break
		 if(i_episode % plot_every == 0):
		 	scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0,num_e 'pisodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
    return Q

# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)])
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RRIGHT = 1, DOWN = 2, LEFT = 3, N?A =-1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)

# Part 2: TD Control: Q-learning(Sarsamax)

# Input: policy π, positive integer num episodes, small positive fraction α, GLIE {εi}
# Output: value function Q (≈ qπ if num episodes is large enough)
# Initialize Q arbitrarily (e.g., Q(s, a) = 0 for all s ∈ S and a ∈ A(s), and Q(terminal-state, ·) = 0)
# for i ← 1 to num episodes do 
# 	ε ← εi
#   Observe S0 
#   t←0 
#   repeat
#      Choose action At using policy derived from Q (e.g., ε-greedy) 
#      Take action At and observe Rt+1 , St+1
#      Q(St, At) ← Q(St, At) + α(Rt+1 + γ maxa Q(St+1, a) − Q(St, At)) 
#      t←t+1
#   until St is terminal; 
# end
# return Q


def  q_learning(env, num_episodes, alpha, gamma=1.0):
	# initialize action-value function (empty dictionary of arrays)
	Q = defaultdict(lambda: np.zeros(env.nA))
	# initialize performance monitor
	plot_every = 100
	tmp_scores = deque(maxlen=plot_every)
	scores = deque(maxlen=num_episodes)
	# loop over episodes
	for i_episode in range(1, num_episodes+1):
		# monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # initialize score
        score = 0
        # begin an episode, observe S
        state = env.reset()
        while True:
        	# get epsilon-greedy action probabilities
        	policy_s = epsilon_greedy_probs(env, Q[state], i_episode)
        	# pick next action A
        	action = np.random.choice(np.arange(env.nA), p=policy_s)
        	# take action A, observe R, S'
        	next_state, reward, done, info = env.step(action)
        	# update Q
        	Q[state][action] = update_Q(Q[state][action], np.max(Q[next_state]), \
        									reward, alpha, gamma)

        	# S <- S'
        	state = next_state
        	# until S is terminal
        	if done:
        		# append score
        		tmp_scores.append(score)
        		break
        if(i_episode % plot_every == 0):
        	scores.append(np.mean(tmp_scores))
    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
    return Q


	# obtain the estimated optimal policy and corresponding action-value function
	Q_sarsamax = q_learning(env, 5000, .01)

	# print the estimated optimal policy
	policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
	check_test.run_check('td_control_check', policy_sarsamax)
	print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
	print(policy_sarsamax)

	# plot the estimated optimal state-value function
	plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])

	#Part 3: TD Control: Expected Sarsa

	# Input: policy π, positive integer num episodes, small positive fraction α, GLIE {εi}
	# Output: value function Q (≈ qπ if num episodes is large enough)
	# Initialize Q arbitrarily (e.g., Q(s, a) = 0 for all s ∈ S and a ∈ A(s), and Q(terminal-state, ·) = 0)
	# for i ← 1 to num episodes do 
	#     ε ← εi
	#     Observe S0 
	#     t←0 
	#     repeat
	#       Choose action At using policy derived from Q (e.g., ε-greedy)
	#       Take action At and observe Rt+1 , St+1
	#       Q(St, At) ← Q(St, At) + α(Rt+1 + γ 􏰀a π(a|St+1)Q(St+1, a) − Q(St, At)) 
	#       t←t+1
	#     until St is terminal; 
	# end
	# return Q

	def expected_sarsa(env, num_episodes, alpha, gamma=1.0):
		# initialize action-value function (empty dictionary of arrays)
		Q = defaultdict(lambda: np.zeros(env.nA))
		# initialize performance moniter
		plot_every = 100
		tmp_scores deque(maxlen=plot_every)
		scores = deque(maxlen=num_episodes)
		# loop over episodes
		for i_episode in range(1, num_episodes+1):
			# monitor progress
			if i_episode % 100 == 0:
				print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
				sys.stdout.flush()
			# initialize score
			socre = 0
			# begin an episode
			state = env.reset()
			# get epsilon-greedy action probabilities
			policy_s = epsilon_greedy_probs(env, Q[state], i_episode, 0.005)
			while True:
				# pick next action
				action = np.random.choice(np.arange(env.nA), p=policy_s)
				# take action A, observe R, S'
				next_state, reward, done, info = env.step(action)
				# add reward to score
				score += reward
				# get epsilon-greedy action probabilities (for S')
				policy_s = epsilon_greedy_probs(env, Q[next_state], i_episode, 0.005)
				# update Q
				Q[state][action] = update_Q(Q[state][action], np.dot(Q[next_state], policy_s), \
													reward, alpha, gamma)

				# S <- S'
				state = next_state
				# until S is terminal
				if done:
					# append score
					tmp_scores.append(score)
					break
	        if (i_episode % plot_every == 0):
	            scores.append(np.mean(tmp_scores))
	    # plot performance
	    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
	    plt.xlabel('Episode Number')
	    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
	    plt.show()
	    # print best 100-episode performance
	    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
	    return Q



# obtain the estimated optimal policy and corresponding action-value function
Q_expsarsa = expected_sarsa(env, 10000, 1)

# print the estimated optimal policy
policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_expsarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expsarsa)

# plot the estimated optimal state-value function
plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])





