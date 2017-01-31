from models.linear_regression import LinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression as sklr

tab = np.loadtxt('lineardata.txt', skiprows=True)
tab = tab[:,1:]

y = tab[:,0]
X = tab[:,1:]

reg = LinearRegression().fit(X,y)
score = reg.score(X,y)
print(score)

skreg = sklr().fit(X,y)
predictions = skreg.predict(X)
print(np.mean((predictions-y)**2))