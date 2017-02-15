import numpy as np
from optimization import AdaptiveGradientDescent

class LinearModel:

	def __init__(self, regularization, optimizer, tol):
		self.regularization = regularization
		self.optimizer = optimizer(epsilon=tol)

	def fit(self, X, y):
		"""Fits the model to the data according to the parameters"""
		pass

	def predict(self, X):
		"""Predicts the target values according to the fitted weights. The "fit" method must have been called beforehand."""
		pass

	def get_params(self):
		"""Returns a dictionnary containing the name and values of the fitted parameters. Have to be called after the fit method."""
		pass



class LinearRegression(LinearModel):
	"""Linear regression model. No regularization, l1 (Lasso), l2(Ridge) and elastic-net are implemented.
		alpha parameter is used for l1 and l2 regularization, rho is the ratio of l1 regularization for elastic-net"""

	def __init__(self, regularization=None, alpha=1.0, rho = 0.5, optimizer=AdaptiveGradientDescent, tol=1e-4):
		assert regularization in [None, 'l1', 'lasso', 'l2', 'ridge', 'elastic-net'], "regularization must be None (default), 'l1', 'lasso', 'l2', 'ridge' or 'elastic-net'"
		super().__init__(regularization, optimizer, tol)
		if self.regularization is not None:
			self.alpha = alpha

			self.pred = lambda X,n,b,w: np.dot(X,w) + b

			mse_loss = lambda X,y,n,b,w: np.dot(y-self.pred(X,n,b,w), y-self.pred(X,n,b,w))/(2*n)
			mse_grad_b = lambda X,y,n,b,w: - np.sum(y-self.pred(X,n,b,w))/n
			mse_grad_w = lambda X,y,n,b,w: - np.dot(X.T, y-self.pred(X,n,b,w))/n

			l2_loss = lambda w: self.alpha*np.dot(w.T, w)/2
			l2_grad = lambda w: self.alpha*w

			l1_loss = lambda w: self.alpha*np.linalg.norm(w, ord=1)
			l1_grad = lambda w: self.alpha*np.sign(w)

		if self.regularization in ['l2', 'ridge']:
			self.loss = lambda X,y,n,b,w: mse_loss(X,y,n,b,w) + l2_loss(w)
			self.grad = lambda X,y,n,b,w: np.hstack((
				mse_grad_b(X,y,n,b,w),
				mse_grad_w(X,y,n,b,w) + l2_grad(w)
				))
		elif self.regularization in ['l1', 'lasso']:
			self.loss = lambda X,y,n,b,w: mse_loss(X,y,n,b,w) + l1_loss(w)
			self.grad = lambda X,y,n,b,w: np.hstack((
				mse_grad_b(X,y,n,b,w),
				mse_grad_w(X,y,n,b,w) + l1_grad(w)
				))
		elif self.regularization == 'elastic-net':
			self.rho = rho
			self.loss = lambda X,y,n,b,w: mse_loss(X,y,n,b,w) + self.rho*l1_loss(w) + (1-self.rho)*l2_loss(w)
			self.grad = lambda X,y,n,b,w: np.hstack((
				mse_grad_b(X,y,n,b,w),
				mse_grad_w(X,y,n,b,w) + l1_grad(w) + l2_grad(w)
				))

	def fit(self, X, y):
		if self.regularization is None:
			X = np.hstack((np.ones((X.shape[0],1)), X)) # takes the bias into account
			#self.w = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))
			# I think this formulation is more efficient
			self.w = np.linalg.solve(np.dot(X.T,X), np.dot(X.T,y))
		else:
			n = len(y)
			pred = lambda b,w: self.pred(X,y,n,b,w)
			loss = lambda b,w: self.loss(X,y,n,b,w)
			grad = lambda b,w: self.grad(X,y,n,b,w)

			v0 = 2*np.random.random_sample((X[0].shape[0]+1)) - 1 # pick random values in [-1,1]
			b0, w0 = v0[0], v0[1:]
			b_min, w_min = self.optimizer.minimize(loss, grad, b0, w0)
			self.b, self.w = b_min, w_min

		return self

	def predict(self, X):
		if self.regularization is None:
			X = np.hstack((np.ones((X.shape[0],1)), X))
			return np.dot(X,self.w)
		else:
			return np.dot(X,self.w) + self.b

	def get_params(self):
		if self.regularization is None:
			return {'w': self.w}
		elif regularization == 'elastic-net':
			return {'w': self.w, 'b': self.b, 'alpha': self.alpha, 'rho': self.rho}
		else:
			return {'w': self.w, 'b': self.b, 'alpha': self.alpha}

	def r2(self, X, y):
		"""Computes the coefficient of determination of the linear regression"""
		predictions = self.predict(X)
		return 1 - np.mean((y-predictions)**2)/y.var()

	def mse(self, X, y):
		"""Computes mean squared error."""
		predictions = self.predict(X)
		return np.mean((predictions-y)**2)


class LogisticRegression(LinearModel):
	"""Implements logistic regression for binary classification.
	C is the coefficient used in front of the learning term of the loss, like in scikit-learn."""

	def __init__(self, regularization='l2', C=1.0, optimizer=AdaptiveGradientDescent, tol=1e-4):
		super().__init__(regularization, optimizer, tol)
		self.C = C
		if regularization == 'l1':
			self.reg_loss = lambda w: np.linalg.norm(w, ord=1)
			self.reg_grad = lambda w: np.sign(w)
		elif regularization == 'l2':
			self.reg_loss = lambda w: np.dot(w.T, w)/2
			self.reg_grad = lambda w: w
		else:
			raise ValueError("regularization must be 'l1' or 'l2'.")


	def fit(self, X, y):
		self.unique = set(y)
		scalar_pred = lambda b,w: np.dot(X,w) + b

		if self.unique == set([-1,1]):
			loss = lambda b,w: self.C*np.log(1+np.exp(-y*scalar_pred(b,w))).sum() + self.reg_loss(w)

			v = lambda b,w: y / (1+np.exp(y*scalar_pred(b,w)))

			grad = lambda b,w: np.hstack((
				-self.C*v(b,w).sum(),
				-self.C*np.dot(v(b,w),X) + self.reg_grad(w)
				))

		elif self.unique == set([0,1]):	# we could simply replace y by 2*y-1 to use the former formulas, but as this library has an educationnal purpose, I prefered implementing both cases
			loss = lambda b,w: self.C*np.sum(y*np.log(1+np.exp(-scalar_pred(b,w))) + (1-y)*np.log(1+np.exp(scalar_pred(b,w)))) + self.reg_loss(w)

			v = lambda b,w: 1/(1+np.exp(-scalar_pred(b,w))) - y

			grad = lambda b,w: np.hstack((
				self.C*v(b,w).sum(),
				self.C*np.dot(v(b,w),X) + self.reg_grad(w)
				))

		elif len(self.unique) == 2:
			raise ValueError('Target must either take values in {-1,1} or {0,1}')
		else:
			raise ValueError('This model is for binary classification only. Target must take two unique values.')

		v0 = 2*np.random.random_sample((X[0].shape[0]+1)) - 1
		b0, w0 = v0[0], v0[1:]
		b_min, w_min = self.optimizer.minimize(loss, grad, b0, w0)
		self.b, self.w = b_min, w_min

		return self


	def predict(self, X, threshold=0.5):
		predictions = (1/(1+np.exp(-(np.dot(X,self.w)+self.b))) >= threshold).astype(np.int)
		if self.unique == set([-1,1]):
			return 2*predictions - 1
		else:
			return predictions

	def accuracy(self, X, y, threshold=0.5):
		predictions = self.predict(X, threshold)
		return np.mean(predictions==y)

	def get_params(self):
		return {'w': self.w, 'b': self.b, 'C': self.C}