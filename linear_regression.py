import numpy as np
from model import Model
from gradient_descent import GradientDescentOptimizer


class LinearRegression(Model):


	def __init__(self, fit_bias=True):
		Model.__init__(self)
		self.fit_bias = fit_bias


	def fit(self, X, y):
		"""Fits the model. For linear regresion, we have an exact formula."""

		if self.fit_bias:
			X = np.hstack((np.ones((X.shape[0],1)), X))

		#self.w = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))

		# I think this formulation is more efficient
		self.w = np.linalg.solve(np.dot(X.T,X), np.dot(X.T,y))
		return self


	def predict(self, X):
		"""Predicts the target values according to the fitted weights. The "fit" method must have been called beforehand."""
		if self.fit_bias:
			X = np.hstack((np.ones((X.shape[0],1)), X))

		return np.dot(X,self.w)

	def score(self, X, y):
		"""Computes the coefficient of determination of the linear regression"""
		predictions = self.predict(X)
		return 1 - np.mean((y-predictions)**2)/y.var()


class RidgeRegression(Model):

	def __init__(self, C=1.0, fit_bias=True):
		Model.__init__(self)
		self.fit_bias = fit_bias
		self.C = C
		self.optimizer = GradientDescentOptimizer()

	def fit(self, X, y):
		if self.fit_bias:
			X = np.hstack((np.ones((X.shape[0],1)), X))
			
		func = lambda w: np.dot(y-np.dot(X, w), y-np.dot(X, w))/2 + self.C*np.dot(w.T, w)/2
		grad = lambda w: self.C*w - np.dot(X.T, y-np.dot(X, w))
		w0 = 2*np.random.random_sample((X[0].shape)) - 1
		w_min = self.optimizer.minimize(func, grad, w0)
		self.w = w_min
		return self

	def predict(self, X):
		"""Predicts the target values according to the fitted weights. The "fit" method must have been called beforehand."""
		if self.fit_bias:
			X = np.hstack((np.ones((X.shape[0],1)), X))

		return np.dot(X,self.w)

	def score(self, X, y):
		"""Computes the coefficient of determination of the linear regression"""
		predictions = self.predict(X)
		return 1 - np.mean((y-predictions)**2)/y.var()