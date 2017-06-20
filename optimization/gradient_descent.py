import numpy as np

from fbml.optimization.base_optimizer import Optimizer

class VanillaGradientDescent(Optimizer):
	"""Vanilla gradient descent. Beware: a learning rate too big can lead to divergence or suboptimial convergence, and too small it can lead to very slow convergence."""

	def __init__(self, gamma=1e-3, epsilon=1e-4):
		super().__init__(gamma, epsilon)


	def minimize(self, func, grad, b0, w0):
		super().minimize(func, grad, b0, w0)
		b, w = b0, w0
		while np.sum(grad(b,w)**2) > self.epsilon:
			step = self.gamma*grad(b,w)
			b = b - step[0]
			w = w - step[1:]

		return b, w


class AdaptiveGradientDescent(Optimizer):
	"""Vanilla gradient descent with adaptive learning rate not to be unbearably inefficient.
		If the loss is decreasing, increases the learning rate by 5%, if it's increasing, halves it and go back to previous point."""

	def __init__(self, gamma=1e-3, epsilon=1e-4):
		super().__init__(gamma, epsilon)


	def minimize(self, func, grad, b0, w0):
		super().minimize(func, grad, b0, w0)
		b, w = b0, w0
		gamma = self.gamma
		old = func(b0,w0)
		while np.sum(grad(b,w)**2) > self.epsilon:
			step = gamma*grad(b,w)
			b = b - step[0]
			w = w - step[1:]

			if func(b,w) < old:
				gamma *= 1.05
			else:
				b = b + step[0]
				w = w + step[1:]
				gamma /=2

			old = func(b,w)

		return b, w


class Momentum(Optimizer):
	"""Implements the gradient descent with momentum algorithm."""

	def __init__(self, gamma=1e-4, momentum=0.9, epsilon=1e-4):
		super().__init__(gamma, epsilon)
		self.momentum = momentum

	def minimize(self, func, grad, b0, w0):
		super().minimize(func, grad, b0, w0)
		b,w = b0, w0
		step = 0
		old = func(b0,w0)
		while np.sum(grad(b,w)**2) > self.epsilon:
			step = self.momentum*step + self.gamma*grad(b,w)
			b = b - step[0]
			w = w - step[1:]
			old = func(b,w)

		return b, w


class Nesterov(Optimizer):
	"""Implements the Nesterov gradient descent algorithm."""

	def __init__(self, gamma=1e-4, momentum=0.9, epsilon=1e-4):
		super().__init__(gamma, epsilon)
		self.momentum = momentum

	def minimize(self, func, grad, b0, w0):
		super().minimize(func, grad, b0, w0)
		b,w = b0, w0
		step = np.zeros(w0.shape[0]+1)
		old = func(b0,w0)
		while np.sum(grad(b,w)**2) > self.epsilon:
			step = self.momentum*step + self.gamma*grad(b - self.momentum*step[0], w - self.momentum*step[1:])
			b = b - step[0]
			w = w - step[1:]
			old = func(b,w)

		return b, w