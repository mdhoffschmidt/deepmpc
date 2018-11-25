#!/usr/bin/env python
# coding=utf-8

# Import third-party packages.
import numpy 
import torch


class Agent:
	"""Agent trainer.
	"""

	BATCH_SIZE = 32
	MIN_SAMPLE = 64
	LR = 0.00025
	L2 = 0.0030

	def __init__(self, model, memory):
		"""Special method for object initialisation.

		:param model: the predictive model.
		:type model: torch module.
		:param memory: replay memory buffer object.
		:type memory: buffer.
		"""

		# Set the model and it loss.
		self.model = model
		self.model_loss = torch.nn.MSELoss()
		self.model_optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.LR, weight_decay=self.L2)

		# Set the memroy.
		self.memory = memory

		# Set the performance.
		self.perf = 0.0
		self.dperf = 0.0

		return

	def load(self, path="agent", model="model"):
		"""Load the model of the agent.

		:param path: the name of the path.
		:type path: str.
		:param model: the name of the model.
		:type model: str.
		"""
		filename = "{}/{}.txt".format(path, model)
		self.model.load_state_dict(torch.load(filename))
		return

	def save(self, path="agent", model="model"):
		"""Save the policy and target networks.

		:param path: the name of the path.
		:type path: str.
		:param model: the name of the model.
		:type model: str.		
		"""
		filename = "{}/{}.txt".format(path, model)
		torch.save(self.model.state_dict(), filename)
		return

	# ------------------------- #
	# ---  0. Model methods --- #
	# ------------------------- #

	def model_parameter_norm(self):
		"""Returns the norm of the model parameters.

		:return: the norm of the parameters.
		:rtype: float.
		"""
		norm = torch.tensor(0.0)
		for param in self.model.parameters():
			norm += torch.norm(param)
		return norm.item()

	def model_performance(self):
		"""Evaluate the performance of the model on the memory.

		:return: the performance for the model.
		:rtype: float.
		"""

		# Set the model in evaluation mode.
		self.model.eval()

		# Get the batch.
		batch = self.memory.batch()
		state, action, next_state = batch

		# Get the next state predictions.
		pred_state = self.model(state, action)

		# Get the performance. 
		error = ( pred_state - next_state )
		perf = torch.mean( error * error )

		return perf.item()

	def optimize(self):
		"""Samples a random batch from replay memory and optimize.

		:return: the loss.
		:rtype: float.		
		"""

		# Set the model in training mode.
		self.model.train()

		# Check the size of the memory.
		if len(self.memory) <= self.MIN_SAMPLE:
			return None

		# Get samples out of the memory.
		batch = self.memory.sample(self.BATCH_SIZE)
		state, action, next_state = batch

		# Get the predictions for the next states.
		pred_state = self.model(state, action)

		# Set the gradients of the optimizer.
		self.model_optimizer.zero_grad()

		# Perform the backward step.
		loss = self.model_loss(pred_state, next_state)
		loss.backward()

		# Perform one optimisation step.
		self.model_optimizer.step()

		return loss.item()
