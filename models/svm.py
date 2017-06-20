import numpy as np

from fbml.optimization.smo import SMOSolver

# WORKS AT LEAST A BIT SINCE THE SCORES ARE BETTER THAN RANDOM, BUT NOT PROPERLY
# OPTIMIZATION IS PROBABLY WRONG SOMEWHERE
# CHECK IF THE INDICES OF THE SUPPORTS ARE NOT SHUFFLED AT SOME POINT

class SVM():
    def __init__(self, C=1.0, kernel='rbf', gamma='auto', degree=3, coef0=0.0, tol=1e-3, epsilon=1e-3):
        
        if kernel not in ['rbf', 'linear', 'poly', 'polynomial', 'sigmoid']:
            raise ValueError("Wrong kernel argument.")
            
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        
        self.tol = tol
        self.epsilon = epsilon
    
    
    def compute_kernel_matrix(self, X):
        n = len(X)
        K = np.zeros((n,n))
        for i,x1 in enumerate(X):
            for j,x2 in enumerate(X):
                K[i,j] = self.kernel_func(x1,x2)
                
        return K
    
    def kernel_func(self, x, z):
        if self.kernel == 'linear':
            return np.dot(x,z)
        elif self.kernel in ['poly', 'polynomial']:
            return (np.dot(x,z) + self.coef0)**self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma*((x-z)**2).sum())
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma*np.dot(x,z) + self.coef0)
        
        
    def fit(self, X, y):
        if not set(y) == set([-1,1]):
            raise ValueError("y must contain only 1 and -1.")
        
        if self.kernel in ['rbf', 'sigmoid'] and self.gamma == 'auto':
            self.gamma = 1./X.shape[1]
            
        K = self.compute_kernel_matrix(X)
        
        if self.kernel == 'linear':
            smo = SMOSolver(X, y, self.C, K, self.tol, self.epsilon, linear=True)
            self.w, self.b = smo.train()
        else:
            smo = SMOSolver(X, y, self.C, K, self.tol, self.epsilon, linear=False)
            support_idx, alphas, self.b = smo.train()
            self.support_vectors = X[support_idx]
            self.support_alphas = alphas[support_idx]
            self.support_y = y[support_idx]
            self.n_support = len(support_idx)
            
        return self
    
    
    def predict(self, X):
        if self.kernel == 'linear':
            pred = np.dot(X, self.w) - self.b
        else:
            pred = []
            for x in X:
                s = -self.b
                for i in range(self.n_support):
                    x_i = self.support_vectors[i]
                    alpha_i = self.support_alphas[i]
                    y_i = self.support_y[i]
                    s += alpha_i*y_i*self.kernel_func(x_i,x)
                    
                pred.append(s)
            pred = np.array(pred)
            
        return np.sign(pred)
    
    def score(self, X, y):
        _y = self.predict(X)
        return (y==_y).mean()