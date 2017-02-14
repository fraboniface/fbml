import numpy as np

from linear_model import LinearRegression
from sklearn.linear_model import LinearRegression as sklr, Ridge, Lasso, ElasticNet

from gradient_descent import VanillaGradientDescent, Momentum, Nesterov

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

if __name__ == '__main__':
	test_linear_regression()