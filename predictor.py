#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Predictor(nn.Module):
	"""Predictor network
	"""

	def __init__(self, state_dim, action_dim, hidden=64):
		"""Special method for class initialisation.

		:param state_dim: Dimension of input state.
		:type state_dim: int.
		:param action_dim: Dimension of input action.
		:type action_dim: int.
		"""
		super(Predictor, self).__init__()

		self.s_dim = state_dim
		self.a_dim = action_dim
		self.hidden = hidden
	
		self.fcs1 = nn.Linear(self.s_dim, self.hidden)
		self.fca1 = nn.Linear(self.a_dim, self.hidden)

		self.fc1 = nn.Linear(2*self.hidden, self.hidden)
		self.fc2 = nn.Linear(self.hidden, self.s_dim)
		
		return

	def forward(self, s, a):
		"""Returns Value function Q(s,a) obtained from critic network.

		:param s: state with size (s_dim, n).
		:type s: torch tensor or numpy array.
		:param a: action with size (a_dim, n)
		:type a: Torch tensor or numpy array.
		"""

		# Check numpy type for a and a.
		FLAG = False
		if isinstance(s, np.ndarray):
			s = torch.Tensor(s)
			FLAG = True
		if isinstance(a, np.ndarray):
			a = torch.Tensor(a)
			FLAG = True

		# Check torch type of s and a.
		if not isinstance(s, torch.Tensor):
			raise TypeError()
		if not isinstance(a, torch.Tensor):
			raise TypeError()

		# Tranpose the state and action.
		s = s.t()
		a = a.t()

		# Perform the forward pass.
		s1 = F.relu(self.fcs1(s))
		a1 = F.relu(self.fca1(a))
		s2 = torch.cat((s1, a1), dim=1)
		
		# Compute the output.
		s3 = F.relu(self.fc1(s2))
		out = self.fc2(s3).t()

		# Check if input was numpy.
		if FLAG is True:
			out = out.detach().numpy()

		return out

	def state_matrix(self, s, a):
		"""Returns the state matrix.

		:param s: state with size (s_dim, 1).
		:type s: numpy array.
		:param a: action with size (a_dim, 1)
		:type a: numpy array.		
		"""
		s = torch.autograd.Variable(torch.Tensor(s), requires_grad=True)
		a = torch.autograd.Variable(torch.Tensor(a), requires_grad=True)		
		out = self.__call__(s, a)
		A = []
		for i in range(self.s_dim):
			out[i].backward(retain_graph=True)
			grad = s.grad.detach().numpy().copy()
			A.append(grad)
			a.grad.zero_()
			s.grad.zero_()
		return np.transpose(np.hstack(A))

	def state_matrix_diff(self, s, a, eps=1.0E-3):
		"""Returns the state matrix.

		:param s: state with size (s_dim, 1).
		:type s: numpy array.
		:param a: action with size (a_dim, 1)
		:type a: numpy array.
		"""

		# Set the disturbation matrix.
		d = eps * np.identity(self.s_dim)

		# Repeat the state and action vectors.
		s = np.tile(s, (1, self.s_dim))
		a = np.tile(a, (1, self.s_dim))

		# Build the state difference matrix.
		A = ( self.forward(s + d, a) - self.forward(s - d, a) ) / ( 2 * eps )

		return A

	def input_matrix(self, s, a):
		"""Returns the input matrix.

		:param s: state with size (s_dim, 1).
		:type s: numpy array.
		:param a: action with size (a_dim, 1)
		:type a: numpy array.		
		"""
		s = torch.autograd.Variable(torch.Tensor(s), requires_grad=True)
		a = torch.autograd.Variable(torch.Tensor(a), requires_grad=True)		
		out = self.__call__(s, a)
		B = []
		for i in range(self.s_dim):
			out[i].backward(retain_graph=True)
			grad = a.grad.detach().numpy().copy()
			B.append(grad)
			a.grad.zero_()
			s.grad.zero_()
		return np.transpose(np.hstack(B))

	def input_matrix_diff(self, s, a, eps=1.0E-3):
		"""Returns the input matrix.

		:param s: state with size (s_dim, 1).
		:type s: numpy array.
		:param a: action with size (a_dim, 1)
		:type a: numpy array.
		"""

		# Set the disturbation matrix.
		d = eps * np.identity(self.a_dim)

		# Repeat the state and action vectors.
		s = np.tile(s, (1, self.a_dim))
		a = np.tile(a, (1, self.a_dim))

		# Build the state difference matrix.
		B = ( self.forward(s, a + d) - self.forward(s, a - d) ) / ( 2 * eps )

		return B

	def linearise(self, s, a, diff=False):
		"""Linearise the model around s, a.

		:param s: state with size (s_dim, 1).
		:type s: numpy array.
		:param a: action with size (a_dim, 1)
		:type a: numpy array.
		:param diff: Use finite difference if True, False othewise.
		:type diff: bool.	
		"""
		if diff:
			A = self.state_matrix_diff(s, a)
			B = self.input_matrix_diff(s, a)
		else:
			A = self.state_matrix(s, a)
			B = self.input_matrix(s, a)
		return (A, B)

	def pred_linear(self, x, u, s, a):
		return
