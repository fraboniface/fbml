import numpy as np
from base_optimizer import Optimizer

class VanillaGradientDescent(Optimizer):

	def __init__(self, gamma=1e-3):
		"""Vanilla gradient descent with adaptive learning rate not to be unbearably inefficient."""
		Optimizer.__init__(self, gamma)


	def minimize(self, func, grad, b0, w0, epsilon=5e-8):
		points = 10*np.random.random_sample((10, w0.shape[0]+1)) - 5	# we pick 10 points in [-5,5]
		assert self.gradient_check(func, grad, points), "Gradient check has failed"

		b, w = b0, w0
		gamma = self.gamma
		old = func(b0,w0)
		diff = np.inf
		while abs(diff) > epsilon:
			step = gamma*grad(b,w)
			b = b - step[0]
			w = w - step[1:]

			diff = func(b,w) - old
			# we use an adaptive learning rate to go fatser: if the loss is decreasing, increase the learning rate by 5%, if it's increasing, halve it and go back to previous point
			if diff < 0:
				gamma *= 1.05
			else:
				b = b + step[0]
				w = w + step[1:]
				gamma /=2

			old = func(b,w)

		return b, w


class Momentum(Optimizer):

	def __init__(self, gamma=1e-4, momentum=0.9):
		Optimizer.__init__(self, gamma)
		self.momentum = momentum

	def minimize(self, func, grad, b0, w0, epsilon=5e-8):
		points = 10*np.random.random_sample((10, w0.shape[0]+1)) - 5	# we pick 10 points in [-5,5]
		assert self.gradient_check(func, grad, points), "Gradient check has failed"

		b,w = b0, w0
		step = 0
		old = func(b0,w0)
		diff = np.inf
		while abs(diff) > epsilon:
			step = self.momentum*step + self.gamma*grad(b,w)
			b = b - step[0]
			w = w - step[1:]
			diff = func(b,w) - old
			old = func(b,w)

		return b, w