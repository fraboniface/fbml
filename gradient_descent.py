import numpy as np
from base_optimizer import Optimizer

class GradientDescentOptimizer(Optimizer):

	def __init__(self, gamma=1e-3):
		Optimizer.__init__(self, gamma)


	def minimize(self, func, grad, w0, epsilon=1e-8):
		points = 10*np.random.random_sample((10, *w0.shape)) - 5	# we pick 10 points in [-5,5]
		assert self.gradient_check(func, grad, points), "Gradient check has failed"

		w = w0
		while np.linalg.norm(grad(w)) > epsilon:
			# NE MARCHE PAS, VOIR S'IL NE FAUDRAIT PAS SÉPARER LE BIAIS DES POIDS !!!
			w = w - self.gamma*grad(w)

		return w