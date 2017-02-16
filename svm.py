import numpy as np
from optimization import AdaptiveGradientDescent

class SVM:
"""Implementation of support vector machine for classification (SVC in scikit-learn)."""

	def __init__(self, C=1.0, kernel='rbf', gamma='auto', degree=3, coef0=0.0, optimizer=AdaptiveGradientDescent, tol=1e-4):
		self.C = C
		self.optimizer = optimizer(epsilon=tol)

		#rbf, linear, polynomial, sigmoid
		if kernel == 'linear':
			self.kernel = lambda u,v: np.dot(u,v)

		elif kernel == 'polynomial':
			self.gamma = gamma
			self.kernel = lambda gamma,u,v: (gamma*np.dot(u,v) + coef0)**degree

		elif kernel == 'rbf':
			assert gamma > 0 or gamma=='auto', "gamma must be greater than 0 for rbf kernel"
			self.gamma = gamma
			self.kernel = lambda gamma,u,v: np.exp(-gamma*(u-v)**2)

		elif kernel == 'sigmoid':
			self.gamma = gamma
			self.kernel = lambda gamma,u,v: np.tanh(gamma*np.dot(u,v) + coef0)

		else:
			raise ValueError("kernel must be 'linear', 'polynomial', 'rbf' or 'sigmoid'.")


	def fit(self, X, y):
		if kernel != 'linear' and self.gamma == 'auto':
			self.gamma = 1/X.shape[1]
			self.kernel = lambda u,v: self.kernel(self.gamma,u,v)

		n = X.shape[0]
		Q = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				Q[i,j] = y[i]*y[j]*self.kernel(X[i], X[j])

		#trouver quoi faire des zeta_i