#!/usr/bin/env python
# coding=utf-8

# Import built-in packages.
import time
import math

# Import third-party packages.
import gym
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch
import cvxpy as cp
import numpy as np

# Import classes.
from agent import Agent
from memory import Memory
from predictor import Predictor

# Set hyper parameters.
EPISODE = 200
MAX_COUNTER = 200
CAPACITY = 100000
LEVEL = -1
MODEL = "model"
AGENT = "agent"

# Set the controller variables.
T = 50
n = 3
m = 1
x = cp.Variable((n, T+1))
u = cp.Variable((m, T))

s0 = np.asarray([[1.0], [0.0], [0.0]])
a0 = np.asarray([[0.0]])
beta = np.asarray([1.0, 1.0, 0.1])

A = cp.Parameter((n, n))
B = cp.Parameter((n, m))
x0 = cp.Parameter((n))
u0 = cp.Parameter((m))

# Loop on the horizon.
cost = 0
constr = []

for t in range(T):

	# Add the cost.
	cost += cp.sum_squares( (x[:,t+1] + x0 - s0[:,0]) * beta ) + cp.sum_squares( 0.001 * u[:,t])

	# Set the constraint.
	constr += [x[:,t+1] == A * x[:,t] + B[:,0] * u[:,t],
	cp.norm(u[:,t] + u0, "inf") <= 2.0]

# Set the initial constraint.
constr += [ x[:,0] == 0]

# Build the problem.
problem = cp.Problem(cp.Minimize(cost), constr)

# -------------------------------------------- #

def controller(x1=None, u1=None, A=None, B=None, new=True):
	"""
	"""

	# Solve the problem.
	problem.solve(warm_start=True)

	return u[:,0].value.reshape(-1,1) #+ u0.reshape(-1,1)

# -------------------------------------------- #

if __name__ == "__main__":

	# Make the gym environment.
	env = gym.make("Pendulum-v0")

	# Get the environment action space.
	action_space = env.action_space.shape[0]
	state_space = env.observation_space.shape[0]

	# Get the memory.
	memory = Memory(CAPACITY)

	# Get the predictor.
	predictor = Predictor(state_space, action_space, hidden=128)

	# Load the trainer.
	agent = Agent(predictor, memory)

	# Load a pretrained model.
	try:
		agent.load(AGENT, MODEL)
	except:
		print("Unable to load the agent.")

	# Load the writer.
	number = int( time.time() % ( 3600 * 24 ))
	writer = SummaryWriter("tensorboard/runs_{}".format(number))

	# Set the global counter.
	global_counter = 0
	verbose = True

	# Perform a number of episodes.
	for i_episode in range(EPISODE):

		# Reset the environment.
		state = env.reset()
		state = state.reshape(-1, 1)

		# Set the done flag.
		done = False

		# Set the initial counter.
		counter = 0
		action = a0

		# Step on the environment while not done.
		while not done:

			# Render the environment.
			env.render()

			# Linearise.
			A_new, B_new = predictor.linearise(state, action, diff=True)

			# Get the controller action.
			if i_episode > LEVEL:
				control_action = True
			else:
				control_action = False

			# Controller.
			if control_action is True:
				t_a = time.time()

				x0.value = state[:,0]
				u0.value = action[:,0]
				A.value = A_new
				B.value = B_new

				#action = controller()

				if counter == 0:
					problem.solve()
				else:
					#print(A.value)
					#print(B.value)
					problem.solve(solver=cp.SCS, warm_start=True)
				print("optimal value with CVXOPT:", problem.value)

				action = (u[:,0].value + u0.value).reshape(-1, 1)

				print(state)
				print(action)

				t_b = time.time()
				print("Elapsed time = {}".format(t_b - t_a))
			else:
				action = env.action_space.sample()
				action = action.reshape(-1, 1)

			# Step on the environment.
			next_state, reward, done, info = env.step(action)
			next_state  = next_state.reshape(-1, 1)

			# Push the results to the memory.
			memory.push(
				torch.Tensor(state), 
				torch.Tensor(action), 
				torch.Tensor(next_state),
				None)

			# Update the next state.
			state = next_state

			# Increment the counter.
			counter += 1
			global_counter += 1

			# Check the counter.
			if counter > MAX_COUNTER:
				break

			# Display.
			if counter % 20 == 0 and verbose is True:
				print("counter = {}".format(counter))
			
			# Train the model.
			if len(memory) > agent.BATCH_SIZE:

				# Get the old perf.
				C_old = agent.model_performance()

				# Perform one optimisation step.
				loss = agent.optimize()

				# Get the new perf.
				C_new = agent.model_performance()

				# Add the memory performance.
				writer.add_scalar("C",
					math.log(C_new / (C_old + 1.0e-8)), 
					global_counter)

				# Add the pnorm.
				writer.add_scalar("pnorm",
					agent.model_parameter_norm(), 
					global_counter)

				# Display the message.
				if loss is not None and verbose is True and counter % 20 == 0:
					msg = []
					msg.append("episode = {}".format(i_episode))
					msg.append("counter = {}".format(counter))
					msg.append("loss = {}".format(loss))
					print(", ".join(msg))

				# Record to the writer.
				if loss is not None and verbose is True:
					writer.add_scalar("loss",
						loss, global_counter)
					writer.add_scalar("log_loss", 
						math.log(loss), global_counter)

		# Save the agent's model.
		agent.save(AGENT, MODEL)

	# Close the environment.
	env.close()

	# ------------- #
	# -- Figure --- #
	# ------------- #

	"""
	fig, ax = plt.subplots(2, sharex=True)
	style = {"linewidth":0.75, "marker":".", "markersize":2.0}
	ax[0].plot(obs[:,0], color="r", **style)
	ax[0].grid()
	ax[1].plot(obs[:,1], color="r", **style)	
	ax[1].grid()
	#ax[2].plot(obs[:,2], color="r", **style)	
	#ax[2].grid()
	#ax[3].plot(obs[:,3], color="r", **style)	
	#ax[3].grid()		
	plt.show()
	plt.close("all")
	"""
