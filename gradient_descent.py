import numpy as np
from base_optimizer import Optimizer

class GradientDescentOptimizer(Optimizer):

	def __init__(self, gamma=1e-3):
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
			if diff < 0:
				gamma *= 1.05
			else:
				b = b + step[0]
				w = w + step[1:]
				gamma /=2

			old = func(b,w)

		return b, w