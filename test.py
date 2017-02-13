import numpy as np

from linear_regression import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LinearRegression as sklr, Ridge as skridge, Lasso as sklasso

from gradient_descent import GradientDescentOptimizer


def test_linear_regression():
	tab = np.loadtxt('lineardata.txt', skiprows=True)
	tab = tab[:,1:]

	y = tab[:,0]
	X = tab[:,1:]

	reg = LinearRegression(regularization='l2').fit(X,y)
	r2 = reg.r2(X,y)
	mse = reg.mse(X,y)
	print('my r2:', r2)
	print('my mse', mse)

	skreg = skridge().fit(X,y)
	print('sklearn r2:', skreg.score(X,y))
	predictions = skreg.predict(X)
	mse =  np.mean((predictions-y)**2)
	print('sklearn mse', mse)


def test_gradient_check():
	opt = GradientDescentOptimizer()
	f = lambda b,w: np.dot(w.T, w) + b
	df = lambda b,w: np.hstack((1, 2*w))
	points = 10*np.random.random_sample((10, 10)) - 5
	return opt.gradient_check(f, df, points)


def test_ridge():
	print('testing ridge')
	tab = np.loadtxt('lineardata.txt', skiprows=True)
	tab = tab[:,1:]

	y = tab[:,0]
	X = tab[:,1:]

	reg = Ridge().fit(X,y)
	r2 = reg.r2(X,y)
	mse = reg.mse(X,y)
	print('my r2:', r2)
	print('my mse', mse)
	print(reg.get_params())

	skreg = skridge().fit(X,y)
	print('sklearn r2:', skreg.score(X,y))
	predictions = skreg.predict(X)
	mse =  np.mean((predictions-y)**2)
	print('sklearn mse', mse)

def test_lasso():
	print('Testing lasso')
	tab = np.loadtxt('lineardata.txt', skiprows=True)
	tab = tab[:,1:]

	y = tab[:,0]
	X = tab[:,1:]

	reg = Lasso().fit(X,y)
	r2 = reg.r2(X,y)
	mse = reg.mse(X,y)
	print('my r2:', r2)
	print('my mse', mse)
	print(reg.get_params())

	skreg = sklasso().fit(X,y)
	print('sklearn r2:', skreg.score(X,y))
	predictions = skreg.predict(X)
	mse =  np.mean((predictions-y)**2)
	print('sklearn mse', mse)



if __name__ == '__main__':
	test_linear_regression()