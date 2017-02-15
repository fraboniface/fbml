import numpy as np
import pandas as pd

from linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import LinearRegression as sklr, Ridge, Lasso, ElasticNet, LogisticRegression as sklr

from optimization import VanillaGradientDescent, Momentum, Nesterov, AdaptiveGradientDescent

from time import time


def test_linear_regression():
	tab = np.loadtxt('lineardata.txt', skiprows=True)
	tab = tab[:,1:]

	y = tab[:,0]
	X = tab[:,1:]

	t0 = time()

	reg = LinearRegression(regularization='l2', optimizer=Nesterov).fit(X,y)
	r2 = reg.r2(X,y)
	mse = reg.mse(X,y)
	print('my r2:', r2)
	print('my mse', mse)

	t1 = time()
	print("My implementation:", t1-t0)

	#skreg = sklr().fit(X,y)
	skreg = Ridge().fit(X,y)
	#skreg = Lasso().fit(X,y)
	#skreg = ElasticNet().fit(X,y)
	print('sklearn r2:', skreg.score(X,y))
	predictions = skreg.predict(X)
	mse =  np.mean((predictions-y)**2)
	print('sklearn mse', mse)

	print("sklearn implementation:", time()-t1)


def test_gradient_check():
	opt = VanillaGradientDescent()
	f = lambda b,w: np.dot(w.T, w) + b
	df = lambda b,w: np.hstack((1, 2*w))
	points = 10*np.random.random_sample((10, 10)) - 5
	return opt.gradient_check(f, df, points)


def test_logistic_regression():
	columns = ['feature'+str(i+1) for i in range(34)]
	columns.append('target')
	df = pd.read_csv('binarydata.csv', header=None, names=columns)
	df.target = (df.target == 'g').astype(np.int)
	X = df.drop(['target'], axis=1).as_matrix()
	y = df.target.as_matrix()
	#y = 2*y - 1

	t0 = time()
	clf  = LogisticRegression(C=100.0, optimizer=Nesterov, tol=1e-4).fit(X,y)
	print('my score', clf.accuracy(X,y))
	t1 = time()
	print("My implementation:", t1-t0)

	skclf  = sklr(C=100.0).fit(X,y)
	print('sklearn score', skclf.score(X,y))
	print("sklearn implementation:", time()-t1)

if __name__ == '__main__':
	test_logistic_regression()