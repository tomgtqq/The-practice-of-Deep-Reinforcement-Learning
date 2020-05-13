import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

# 1. Import the Necessary Package
# Set plotting options
%matplotlib inline
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

!python - m pip install pyvirtualdisplay
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# 2.Specify the Environment, and Explore the State and Action Spaces
# Create an environment and set random seed
env = gym.make('MountainCar-v0')
env.seed(505)


# Watch a random agent
state = env.reset()
img = plt.imshow(env.render(mode='rgb_array'))
for t in range(1000):
    action = env.action_space.sample()
    img.set_data(env.render(mode='rgb_array'))
    plt.axis('off')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    state, reward, done, _ = env.step(action)
    if done:
        print('Score: ', t + 1)
        break

env.close()

# Explore state (observation) space
print("State space", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

# Generate some samples from the state space
print("State space samples:")
print(np.array([env.observation_space.sample() for i in range(10)]))

# Explore the action space
print("Action space:", env.action_space)

# Generate some samples from the action space
print("Action space samples:")
print(np.array([env.action_space.sample() for i in range(10)]))

# 3. Discretize the State Space with a Uniform Grid


def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.

    Parameters
    ----------
    low : array_like
    Lower bounds for each dimension of the continuous space.
    high : array_like
    Upper bounds for each dimension of the continuous space.
    bins : tuple
    Number of bins along each corresponding dimension.

    Returns
    -------
    grid : list of array_like
    A list of arrays containing split points for each dimension.
    """
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1]
            for dim in range(len(bins))]

    return grid


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    # TODO: Implement this
    # apply along each dimension
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))


# 4. Visualization
# It might be helpful to visualize the original and discretized samples to get a sense of how much error you are inroduction
import matplotlib.collections as mc


def visualize_samples(samples, discretized_samples, grid, low=None, high=None):
    """Visualize original and discretized samples on a given 2-dimensional grid."""

    fig, ax = plt.subplots(figsize=(10, 10))

    # Show grid
    ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
    ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
    ax.grid(True)

    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Otherwise use first, last grid locations as low, high (for further mapping discretized samples)
        low = [splits[0] for splits in grid]
        high = [splits[-1] for splits in grid]

    # Map each discretized sample (which is really an index) to the center of corresponding grid cell
    # add low and high ends
    grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))
    # compute center of each grid cell
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2
    locs = np.stack(grid_centers[i, discretized_samples[:, i]]
                    for i in range(len(grid))).T  # map discretized samples

    ax.plot(samples[:, 0], samples[:, 1], 'o')  # plot original samples
    # plot discretized samples in mapped locations
    ax.plot(locs[:, 0], locs[:, 1], 's')
    # add a line connecting each original-discretized sample
    ax.add_collection(mc.LineCollection(
        list(zip(samples, locs)), colors='orange'))
    ax.legend(['original', 'discretized'])


# Create a grid to discretize the state space
state_grid = create_uniform_grid(
    env.observation_space.low, env.observation_space.high, bins=(10, 10))

# Obtain some samples from the space, discretize them, and then visualize them
state_samples = np.array([env.observation_space.sample() for i in range(10)])
discretized_state_samples = np.array(
    [discretize(sample, state_grid) for sample in state_samples])
visualize_samples(state_samples, discretized_state_samples, state_grid,
                  env.observation_space.low, env.observation_space.high)
plt.xlabel('position')
plt.ylabel('velocity')

# Create a grid to discretize the state space
state_grid = create_uniform_grid(
    env.observation_space.low, env.observation_space.high, bins=(10, 10))

# Obtain some samples from the space, discretize them ,and then visualize them
state_samples = np.array([env.observation_space.sample() for i in range(10)])
discretize_state_samples = np.array([discretize(sample, state_grid) fro sample in state_samples])
visualize_samples(state_samples, discretized_state_samples, state_grid,
                  env.observation_space.low, env.observation_space.high)

# axis labels for MountainCar-v0 state space
plt.xlabel('position'); plt.ylabel('velocity');


class QLearningAgent:
	"""Q-Learning agent that can act on a continuous state space by discretizing it."""

	def __init__(self, env, state_grid, alpha=0.02, gamma=0.99,
		epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
		# Environment info
		self.env = env
		self.state_grid = state_grid
		self.state_size = tuple(len(splits) + 1) for splits in self.state_grid)
		self.action_size=self.env.action_space.n
		self.seed=np.random.seed(seed)

		# Learning parameters
		self.alpha=alpha  # learning rate
		self.gamma=gamma  # discount factor
		self.epsilon=self.initial_epsilon=epsilon  # initial exploration rate
		# how quickly should we decrease epsilon
		self.epsilon_decay_rate=epsilon_decay_rate
		self.min_epsilon=min_epsilon

		# Create Q-table
		self.q_table=np.zeros(shape = (self.state_size + (self.action_size, )))

	def preprocess_state(self, state):
		return tuple(discretize(state, self.state_grid))

	def reset_episode(self, state):
		# Gradually decrease exploration rate
		self.epsilon *= self.epsilon_decay_rate
		self.epsilon=max(self.epsilon, self.min_epsilon)

		# Decide initial action
		self.last_state=self.preprocess_state(state)
		self.last_action=np.argmax(self.q_table[self.last_state])
		return self.last_action

    def reset_exploration(self, epsilon = None):
        """Reset exploration rate used when training."""
        self.epsilon=epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward = None, done = None, mode = 'train'):
    	"""Pick next action and update internal Q table (when mode != 'test')."""
    	state=self.preprocess_state(state)
    	if mode == 'test':
    		# Test mode: Simply produce an action
    		action=np.argmax(self.q_table[state])
    	else:
    		# Train mode (default): Update Q table, pick next action
            # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
            	(reward + self.gamma * max(self.q_table[state]) - \
            	 self.q_table[self.last_state + (self.last_action)])

            # Exploration vs. exploitation
            do_exploration=np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
            	# Pick a random action
            	action=np.random.randint(0, self.aciton_size)
            else:
            	# Pick the best action from Q table
            	action=np.argmax(self.q_table[state])
        # Roll over current state, action for next step
        self.last_state=state
        self.last_action=action
        return
   	q_agent=QLearningAgent(env, state_grid)


   	def run(agent, env, num_episodes = 20000, mode = 'train'):
    """Run agent in given reinforcement learning environment and return scores."""
	    scores=[]
	    max_avg_score=-np.inf
	    for i_episode in range(1, num_episodes + 1):
	        # Initialize episode
	        state=env.reset()
	        action=agent.reset_episode(state)
	        total_reward=0
	        done=False

	        # Roll out steps until done
	        while not done:
	            state, reward, done, info=env.step(action)
	            total_reward += reward
	            action=agent.act(state, reward, done, mode)

	        # Save final score
	        scores.append(total_reward)

	        # Print episode stats
	        if mode == 'train':
	            if len(scores) > 100:
	                avg_score=np.mean(scores[-100:])
	                if avg_score > max_avg_score:
	                    max_avg_score=avg_score

	            if i_episode % 100 == 0:
	                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode,
	                      num_episodes, max_avg_score), end = "")
	                sys.stdout.flush()

	    return scores

	scores=run(q_agent, env)

	# Plot scores obtained per episode
	plt.plot(scores); plt.title("Scores")

	def plot_scores(scores, rolling_window = 100):
		"""Plot scores and optional rolling mean using specified window."""
		plt.plot(scores); plt.title("Scores");
		rolling_mean=pd.Series(scores).rolling(rolling_window).mean()
		plt.plot(rolling_mean);
		return rolling_mean

	rolling_mean=plot_scores(scores)

	# Run in test mode and analyze socres obtained
	test_scores=run(q_agent, env, num_episodes = 100, mode = 'test')
	print("[TEST] Completed {} episodes with avg. score = {}".format(
	    len(test_scores), np.mean(test_scores)))
	_=plot_scores(test_scores, rolling_window = 10)


	def plot_q_table(q_table):
    """Visualize max Q-value for each state and corresponding action."""
	    q_image=np.max(q_table, axis = 2)       # max Q-value for each state
	    q_actions=np.argmax(q_table, axis = 2)  # best action for each state

	    fig, ax=plt.subplots(figsize = (10, 10))
	    cax=ax.imshow(q_image, cmap = 'jet');
	    cbar=fig.colorbar(cax)
	    for x in range(q_image.shape[0]):
	        for y in range(q_image.shape[1]):
	            ax.text(x, y, q_actions[x, y], color = 'white',
	                    horizontalalignment = 'center', verticalalignment = 'center')
	    ax.grid(False)
	    ax.set_title("Q-table, size: {}".format(q_table.shape))
	    ax.set_xlabel('position')
	    ax.set_ylabel('velocity')


	plot_q_table(q_agent.q_table)


	state_grid_new=create_uniform_grid(
	    env.observation_space.low, env.observation_space.high, bins = (20, 20))
	q_agent_new=QLearningAgent(env, state_grid_new)
	q_agent_new.scores=[]


	q_agent_new.scores += run(q_agent_new, env,
	                          num_episodes = 50000)  # accumulate scores
	rolling_mean_new=plot_scores(q_agent_new.scores)

	test_scores= run(q_agent_new, env, num_episodes = 100, mode = 'test')
	print("[TEST] Completed {} episodes with avg. score = {}".format(
	    len(test_scores), np.mean(test_scores)))
	_=plot_scores(test_scores)

	plot_q_table(q_agent_new.q_table)

	state=env.reset()
	score=0
	img=plt.imshow(env.render(mode='rgb_array'))
	for t in range(1000):
		action=q_agent_new.act(state, mode = 'test')
		img.set_data(env.render(mode='rgb_array'))
		plt.axis('off')
		display.display(plt.gcf())
		display.clear_output(wait = True)
		state, reward, done, _=env.step(action)
		socre += reward
		if done:
			print('Score: ', socre)
			break
	env.close()
