import numpy as np
from optimization import AdaptiveGradientDescent

class SVM:
"""Implementation of support vector machine for classification (SVC in scikit-learn)."""

	def __init__(self, C=1.0, kernel='rbf', gamma='auto', degree=3, coef0=0.0, optimizer=AdaptiveGradientDescent, tol=1e-4):
		self.C = C
		self.optimizer = optimizer(epsilon=tol)

		#rbf, linear, polynomial, sigmoid
		if kernel == 'linear':
			self.kernel = lambda u,v: np.dot(u,v)

		elif kernel == 'polynomial':
			self.gamma = gamma
			self.kernel = lambda gamma,u,v: (gamma*np.dot(u,v) + coef0)**degree

		elif kernel == 'rbf':
			if gamma != 'auto':
				if gamma < 0:
					raise ValueError("gamma must be 'auto' or greater than 0 for rbf kernel")
			self.gamma = gamma
			self.kernel = lambda gamma,u,v: np.exp(-gamma*(u-v)**2)

		elif kernel == 'sigmoid':
			self.gamma = gamma
			self.kernel = lambda gamma,u,v: np.tanh(gamma*np.dot(u,v) + coef0)

		else:
			raise ValueError("kernel must be 'linear', 'polynomial', 'rbf' or 'sigmoid'.")


	def fit(self, X, y):
		# see if I can simply do it with hinge loss and gradient descent
		if kernel != 'linear':
			if self.gamma == 'auto':
				self.gamma = 1/X.shape[1]
			self.kernel = lambda u,v: self.kernel(self.gamma,u,v)

		n = X.shape[0]
		Q = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				Q[i,j] = y[i]*y[j]*self.kernel(X[i], X[j])

		self.objective = lambda alphas: np.dot(alpha.T, np.dot(Q,alpha))/2 - alpha.sum()
		self.lagrange_multipliers = self.minimize(X, y, n)


	def minimize(self, X, y, n_examples):
		lagrange_multipliers = np.zeros(n_examples)
		numChanged = 0
		examineAll = 1
		while numChanged == 0 or examineAll:
			numChanged = 0
			if examineAll:
				for i in range(n_examples):
					numChanged += examineExample(i)
			else:
				midalphas = [i for i,alpha in enumerate(lagrange_multipliers) if alpha not in [0.0, C]]
				for i in midalphas:
					numChanged += examineExample(i, midalphas)

			if examineAll == 1:
				examineAll = 0
			elif numChanged == 0:
				examineAll = 1

	# STOCKER LES ERREURS, VOIR LEURS TRUC DE THRESHOLD ET DE WEIGHT, FAIRE LA FONCTION DE PRÉDICTION, VOIR SI ON PEUT UTILISER LE MÊME MIDALPHAS DANS TOUTE LES FONCTIONS


	def take_step(self, i1, i2, X, y, epsilon=1e-3):
		if i1 == i2:
			return 0

		alpha1 = self.lagrange_multipliers[i1]
		alpha2 = self.lagrange_multipliers[i2]
		y1 = y[i1]
		y2 = y[i2]
		E1 = SVM_OUT(i1) - y1
		E2 = SVM_OUT(i2) - y2
		s = y1*y2

		if y1 == y2:
			L = max(0, alpha1 + alpha2 - C)
			H = min(C, alpha1 + alpha2)
		else:
			L = max(0, alpha2 - alpha1)
			H = min(C, C + alpha2 - alpha1)

		if L == H:
			return 0

		k11 = self.kernel(X[i1], X[i1])
		k12 = self.kernel(X[i1], X[i2])
		k22 = self.kernel(X[i2], X[i2])
		eta = k11 + k22 - 2*k12

		if eta > 0:
			a2 = alpha2 + y2*(E1-E2)/eta
			if a2 < L:
				a2 = L
			elif a2 > H:
				a2 = H
		else:
			def modify_multipliers(index, value):
				tmp = self.lagrange_multipliers
				tmp[index] = value
				return tmp

			Lobj = self.objective(modify_multipliers(i2, L))
			Hobj = self.objective(modify_multipliers(i2, H))
			if Lobj < Hobj - epsilon:
				a2 = L
			elif Lobj > Hobj + epsilon:
				a2 = H
			else:
				a2 = alpha2

		if abs(a2-alpha2) < epsilon*(a2+alpha2+epsilon):
			return 0

		a1 = alpha1 + s*(alpha2-a2)
		self.lagrange_multipliers[i1] = a1
		self.lagrange_multipliers[i2] = a2

		return 1


	def examine_example(self, X, y, i2, midalphas):
		y2 = y[i2]
		alpha2 = self.lagrange_multipliers[i2]
		E2 = SVM_OUT(i2) - y2
		r2 = E2*y2

		if (r2<-tol and alpha2<self.C) or (r2>tol and alpha2>0):
			if len(midalphas) > 1:
				i1 = second_choice_heuristic(i2)
				if takeStep(i1,i2):
					return 1

			if 




 target = desired output vector
 point = training point matrix
 procedure takeStep(i1,i2)
   Update threshold to reflect change in Lagrange multipliers
   Update weight vector to reflect change in a1 & a2, if SVM is linear
   Update error cache using new Lagrange multipliers
   Store a1 in the alpha array
   Store a2 in the alpha array
   return 1
endprocedure
procedure examineExample(i2)
   y2 = target[i2]
   alph2 = Lagrange multiplier for i2
   E2 = SVM output on point[i2] – y2 (check in error cache)
   r2 = E2*y2
   if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0))
   {
     if (number of non-zero & non-C alpha > 1)
     {
       i1 = result of second choice heuristic (section 2.2)
       if takeStep(i1,i2)
         return 1
     }
     loop over all non-zero and non-C alpha, starting at a random point
     {
        i1 = identity of current alpha
        if takeStep(i1,i2)
          return 1
     }
     loop over all possible i1, starting at a random point
     {
        i1 = loop variable
        if (takeStep(i1,i2)
          return 1
     }
   }
   return 0
endprocedure


""" Sequential Minimal Optimization algorithm for quadratic programming
 target = desired output vector
 point = training point matrix
 procedure takeStep(i1,i2)
   if (i1 == i2) return 0
   alph1 = Lagrange multiplier for i1
   y1 = target[i1]
   E1 = SVM output on point[i1] – y1 (check in error cache)
   s = y1*y2
   Compute L, H via equations (13) and (14)
   if (L == H)
     return 0
   k11 = kernel(point[i1],point[i1])
   k12 = kernel(point[i1],point[i2])
   k22 = kernel(point[i2],point[i2])
   eta = k11+k22-2*k12
   if (eta > 0)
   {
      a2 = alph2 + y2*(E1-E2)/eta
      if (a2 < L) a2 = L
      else if (a2 > H) a2 = H
   }
   else
   {
      Lobj = objective function at a2=L
      Hobj = objective function at a2=H
      if (Lobj < Hobj-eps)
         a2 = L
      else if (Lobj > Hobj+eps)
         a2 = H
      else
         a2 = alph2
   }
   if (|a2-alph2| < eps*(a2+alph2+eps))
      return 0
   a1 = alph1+s*(alph2-a2)
   Update threshold to reflect change in Lagrange multipliers
   Update weight vector to reflect change in a1 & a2, if SVM is linear
   Update error cache using new Lagrange multipliers
   Store a1 in the alpha array
   Store a2 in the alpha array
   return 1
endprocedure
procedure examineExample(i2)
   y2 = target[i2]
   alph2 = Lagrange multiplier for i2
   E2 = SVM output on point[i2] – y2 (check in error cache)
   r2 = E2*y2
   if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0))
   {
     if (number of non-zero & non-C alpha > 1)
     {
       i1 = result of second choice heuristic (section 2.2)
       if takeStep(i1,i2)
         return 1
     }
11
     loop over all non-zero and non-C alpha, starting at a random point
     {
        i1 = identity of current alpha
        if takeStep(i1,i2)
          return 1
     }
     loop over all possible i1, starting at a random point
     {
        i1 = loop variable
        if (takeStep(i1,i2)
          return 1
     }
   }
   return 0
endprocedure
main routine:
    numChanged = 0;
    examineAll = 1;
    while (numChanged > 0 | examineAll)
    {
       numChanged = 0;
       if (examineAll)
          loop I over all training examples
             numChanged += examineExample(I)
       else
          loop I over examples where alpha is not 0 & not C
             numChanged += examineExample(I)
       if (examineAll == 1)
          examineAll = 0
       else if (numChanged == 0)
          examineAll = 1
   }
   """