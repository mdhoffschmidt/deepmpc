#!/usr/bin/env python
# coding=utf-8

# Import third-party packages.
import gym
import time
import math
import numpy
import torch
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# Import classes.
from agent import Agent
from memory import Memory
from predictor import Predictor

# Set hyper parameters.
EPISODE = 200
MAX_COUNTER = 200
CAPACITY = 100000

def test_linearise(predictor, state, action):
	"""Test the linearisation process.
	"""
	A, B = predictor.linearise(state, action)
	A_diff, B_diff = predictor.linearise(state, action, diff=True)
	print("Linearise test")
	print("A error = {}".format(numpy.linalg.norm(A - A_diff)))
	print("B error = {}".format(numpy.linalg.norm(B - B_diff)))
	return


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
	trainer = Trainer(predictor, memory)

	# Get the state and action.
	state = env.reset().reshape(-1, 1)
	action = env.action_space.sample().reshape(-1, 1)


	test_linearise(predictor, state, action)

