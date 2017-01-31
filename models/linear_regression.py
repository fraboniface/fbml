import numpy as np
from models.model import Model


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
		"""Computes the mean squared error of the predicted values against the true values."""
		predictions = self.predict(X)
		return np.mean((predictions-y)**2)