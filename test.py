import numpy as np

from linear_regression import LinearRegression, RidgeRegression
from sklearn.linear_model import LinearRegression as sklr, Ridge as skridge

from gradient_descent import GradientDescentOptimizer


def test_linear_regression():
	tab = np.loadtxt('lineardata.txt', skiprows=True)
	tab = tab[:,1:]

	y = tab[:,0]
	X = tab[:,1:]

	reg = LinearRegression().fit(X,y)
	score = reg.score(X,y)
	print(score)

	skreg = sklr(alpha=0.5).fit(X,y)
	print(skreg.score(X,y))


def test_gradient_check():
	opt = GradientDescentOptimizer()
	f = lambda x: np.dot(x.T, x)
	df = lambda x: 2*x
	points = 10*np.random.random_sample((10, 10)) - 5
	return opt.gradient_check(f, df, points)


def test_ridge():
	tab = np.loadtxt('lineardata.txt', skiprows=True)
	tab = tab[:,1:]

	y = tab[:,0]
	X = tab[:,1:]

	reg = RidgeRegression().fit(X,y)
	score = reg.score(X,y)
	print('my score:', score)

	skreg = skridge().fit(X,y)
	print('sklearn score:', skreg.score(X,y))


if __name__ == '__main__':
	test_ridge()