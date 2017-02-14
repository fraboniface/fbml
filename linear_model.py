import numpy as np
from gradient_descent import Nesterov

class LinearModel:

	def __init__(self, regularization, C, optimizer=VanillaGradientDescent):
		self.regularization = regularization
		self.C = C
		self.optimizer = optimizer()

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
		C parameter is used for l2 regularization and alpha is used for l1 regularization, be these mixed or not."""

	def __init__(self, regularization=None, C=1.0, alpha=1.0, optimizer=Nesterov):
		assert regularization in [None, 'l1', 'lasso', 'l2', 'ridge', 'elastic-net'], "regularization must be None (default), 'l1', 'lasso', 'l2', 'ridge' or 'elastic-net'"
		super().__init__(regularization, C, alpha, optimizer)
		self.alpha = alpha
		if self.regularization is not None:
			self.pred = lambda X,n,b,w: np.dot(X,w) + b*np.ones(n)

			mse_loss = lambda X,y,n,b,w: np.dot(y-self.pred(X,n,b,w), y-self.pred(X,n,b,w))/(2*n)
			mse_grad_b = lambda X,y,n,b,w: - np.sum(y-self.pred(X,n,b,w))/n
			mse_grad_w = lambda X,y,n,b,w: - np.dot(X.T, y-self.pred(X,n,b,w))/n

			l2_loss = lambda w: self.C*np.dot(w.T, w)/2
			l2_grad = lambda w: self.C*w

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
			self.loss = lambda X,y,n,b,w: mse_loss(X,y,n,b,w) + l1_loss(w) + l2_loss(w)
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
			return np.dot(X,self.w) + self.b*np.ones(X.shape[0])

	def get_params(self):
		if self.regularization is None:
			return {'w': self.w}
		else:
			return {'w': self.w, 'b': self.b}

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
	l1 and l2 regularization both use C parameter"""

	def __init__(self, regularization='l2', C=1.0, optimizer=Nesterov):
		super().__init__(regularization, C, alpha, optimizer)
		assert regularization in ['l1', 'l2'], "regularization must be 'l1' or 'l2'."
		if regularization == 'l1':
			reg_loss = lambda w: self.C*np.linalg.norm(w, ord=1)
			reg_grad = lambda w: self.C*np.sign(w)
		elif regularization == 'l2':
			reg_loss = lambda w: self.C*np.dot(w.T, w)/2
			reg_grad = lambda w: self.C*w
		else:
			raise ValueError("regularization must be 'l1' or 'l2'.")


	def fit(self, X, y):
		unique = set(y)
		n = len(y)
		scalar_pred = lambda b,w: np.dot(X,w) + b*np.ones(n)

		if unique == set([-1,1]):
			loss = lambda b,w: np.log(1+np.exp(-y*scalar_pred(b,w))).sum()/n + reg_loss(w)

			v = lambda b,w: y / (1+np.exp(y*scalar_pred(b,w)))

			grad = lambda b,w: np.hstack((
				-v(b,w).sum()/n,
				-np.dot(X,v(b,w))/n + reg_grad(w)
				))

		elif unique == set([0,1]):
			loss = lambda b,w: np.sum(y*np.log(1+np.exp(-scalar_pred(b,w))) + (1-y)*np.log(1+np.exp(scalar_pred(b,w))))/n + reg_loss(w)

			v = lambda b,w: 1/(1+np.exp(-pred(b,w))) - y

			grad = lambda b,w: np.hstack((
				v(b,w).sum()/n,
				np.dot(X,v(b,w))/n + reg_grad(w)
				))

		elif len(unique) == 2:
			raise ValueError('Target must either take values {-1,1} or {0,1}')
		else:
			raise ValueError('This model is for binary classification only. Target must take two unique values.')

		v0 = 2*np.random.random_sample((X[0].shape[0]+1)) - 1 # pick random values in [-1,1]
		b0, w0 = v0[0], v0[1:]
		b_min, w_min = self.optimizer.minimize(loss, grad, b0, w0)
		self.b, self.w = b_min, w_min

		return self