import gym
import numpy as np
import matplotlib.pyplot as plt


def run_episode(env, parameters):
	observation = env.reset()
	totalreward = 0
	for _ in xrange(200):
		action = 0 if np.matmul(parameters, observation) < 0 else 1
		observation, reward, done, info=env.step(action)
		totalreward += reward
		if done:
			break

	return totalreward


def train(submit):
	env = gym.make('CartPole-v0')
	if submit:
		env.monitor.start('cartpole-experiments/', force=True)

	counter = 0
	bestparams = None
	bestreward = 0
	noise_scaling = 0.1
	parameters = np.random.rand(4)*2 - 1

	for _ in xrange(10000):

		counter += 1
		newparams = parameters + (np.random.rand(4)*2 - 1)*noise_scaling
		reward = 0
		run = run_episode(env, parameters) 
		if reward > bestreward:
			bestreward = reward
			parameters = newparams
			if reward == 200:
				break

	if submit:
		for _ in xrange(100):
			run_episode(env, bestparams)

	return counter

results = []

for _ in xrange(1000):
	results.append(train(submit=False))

plt.hist(results, 50, normed=1, facecolor='g', alpha=0.75)
plt.show()