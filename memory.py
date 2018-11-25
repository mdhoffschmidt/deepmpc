#!/usr/bin/env python
# coding=utf-8

import torch
import random
from collections import namedtuple


class Memory(object):
	"""Class that handles a replay memory.

	:attr capacity: the memory capacity.
	:type capacity: int.
	:attr memory: the samples memory.
	:type memory: list.
	:attr position: the memory position.
	:type position: int.
	"""

	Transition = namedtuple('Transition', 
		('state', 'action', 'next_state', 'reward'))	

	def __init__(self, capacity):
		"""Special method for class object construction.
		"""
		self.capacity = capacity
		self.memory = []
		self.position = 0
		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		_repr = {"name":self.__class__.__name__}
		_repr.update({"capacity":self.capacity})
		_repr.update({"position":self.position})
		return _repr

	def __str__(self):
		"""Special methof for class object representation.
		"""
		msg = ["{}: {}".format(a, b) for a, b in self.__repr__().items()]
		return "\n".join(msg)

	def __len__(self):
		"""Special method for class object length.
		"""
		return len(self.memory)

	def push(self, *transition):
		"""Save a transition.

		:param transition: the state, reward, action transition to store.
		:type transition: list of torch tensors.
		"""

		# Check the size of the memory.
		if len(self.memory) < self.capacity:
			self.memory.append(None)

		# Update the memory with the transition.
		self.memory[self.position] = self.Transition(*transition)

		# Update the current pisition.
		self.position = (self.position + 1) % self.capacity

		return

	def batch(self):
		"""Returns the memory batch.
		"""

		# Handle the transition.
		batch = self.Transition(*zip(*self.memory))

		# Handle transition size.
		state = torch.cat(batch.state, dim=1)
		action = torch.cat(batch.action, dim=1)
		next_state = torch.cat(batch.next_state, dim=1)

		return state, action, next_state

	def sample(self, batch_size):
		"""Random samples from the memory.

		:param batch_size: the size of the batch.
		:type batch_size: int.
		"""

		# Check the size of the memeory.
		if self.__len__() < batch_size:
			batch_size = self.__len__()

		# Sample random transitions fro the memory.
		transitions = random.sample(self.memory, batch_size)

		# Handle the transitions.
		batch = self.Transition(*zip(*transitions))

		# Handle transition size.
		state = torch.cat(batch.state, dim=1)
		action = torch.cat(batch.action, dim=1)
		next_state = torch.cat(batch.next_state, dim=1)

		return state, action, next_state

