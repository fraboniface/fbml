import numpy as np

class SMOSolver():
    """
    This is a retranscription of the pseudo-code given in the original SMO paper (Platt, 1998).
    """
    
    def __init__(self, X, y, C, K, tol, epsilon, linear=False):
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        self.C = C
        self.K = K
        
        self.tol = tol
        self.epsilon = epsilon
        
        self.alphas = np.zeros(self.n) # try C/2?
        self.b = 0.0
        
        # to modify if I change alpha initialization
        self.errors = -self.y
        self.non_bound_idx = np.array([], dtype=int)
        
        self.linear = linear
        if linear:
            self.w = np.zeros(self.p)
    
    def train(self):
        num_changed = 0
        examine_all = 1
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self.n):
                    num_changed += self.examine_example(i)
            else:
                for i in self.non_bound_idx:
                    num_changed += self.examine_example(i)
            
            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1
                
        if self.linear:
            return self.w, self.b
        else:
            return self.non_bound_idx, self.alphas, self.b
        
    def second_choice_heuristic(self, E2):
        if E2 >= 0:
            m = np.inf
            for i in self.non_bound_idx:
                if self.errors[i] < m:
                    m = self.errors[i]
                    i1 = i
        else:
            M = -np.inf
            for i in self.non_bound_idx:
                if self.errors[i] > M:
                    M = self.errors[i]
                    i1 = i
                    
        return i1
    
    def objective_function(self, y1, y2, E1, E2, alpha1, alpha2, k11, k12, k22, L, H):
        s = y1*y2
        f1 = y1*(E1+self.b) - alpha1*k11 - s*alpha2*k12
        f2 = y2*(E2+self.b) - s*alpha1*k12 - alpha2*k22
        L1 = alpha1 + s*(alpha2 - L)
        H1 = alpha1 + s*(alpha2 - H)
        psiL = L1*f1 + L*f2 + (L1**2)*k11/2 + (L**2)*k22/2 + s*L*L1*k12
        psiH = H1*f1 + H*f2 + (H1**2)*k11/2 + (H**2)*k22/2 + s*H*H1*k12
        
        return psiL, psiH
        
    def examine_example(self, i2):
        y2 = self.y[i2]
        alpha2 = self.alphas[i2]
        E2 = self.errors[i2]
        r2 = E2*y2
        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            if len(self.non_bound_idx) > 1:
                i1 = self.second_choice_heuristic(E2)
                if self.take_step(i1, i2):
                    return 1
            
            if len(self.non_bound_idx) > 0:
                rd_idx = np.random.randint(0, len(self.non_bound_idx))
                for i1 in np.roll(self.non_bound_idx, rd_idx):
                    if self.take_step(i1, i2):
                        return 1
                
            rd_idx = np.random.randint(0, self.n)
            for i1 in np.roll(np.arange(self.n), rd_idx):
                if self.take_step(i1, i2):
                    return 1
                
        return 0
    
    def take_step(self, i1, i2):
        if i1==i2:
            return 0
        
        alpha1 = self.alphas[i1]
        y1 = self.y[i1]
        E1 = self.errors[i1]
        
        alpha2 = self.alphas[i2]
        y2 = self.y[i2]
        E2 = self.errors[i2]
        
        s = y1*y2
        
        if s == 1:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
            
        if L == H:
            return 0
        
        k11 = self.K[i1,i1]
        k22 = self.K[i2,i2]
        k12 = self.K[i1,i2]
        eta = k11 + k22 - 2*k12
                
        if eta > 0:
            a2 = alpha2 + y2*(E1-E2)/eta
            a2 = max(a2, L)
            a2 = min(a2, H)
        else:
            Lobj, Hobj = self.objective_function(y1, y2, E1, E2, alpha1, alpha2, k11, k12, k22, L, H)
            if Lobj < Hobj - self.epsilon:
                a2 = L
            elif Lobj > Hobj + self.epsilon:
                a2 = H
            else:
                a2 = alpha2
                
        if abs(a2 - alpha2) < self.epsilon*(a2 + alpha2 + self.epsilon):
            return 0
        
        a1 = alpha1 + s*(alpha2-a2)
        
        b1 = E1 + y1*(a1-alpha1)*k11 + y2*(a2-alpha2)*k12 + self.b
        b2 = E2 + y1*(a1-alpha1)*k12 + y2*(a2-alpha2)*k22 + self.b
        new_b = (b1+b2)/2
            
        if self.linear:
            self.w += y1*(a1-alpha1)*X[i1] + y2*(a2 - alpha2)*X[i2]
        
        for i in range(self.n):
            self.errors[i] += (a1 - alpha1)*y1*self.K[i1,i] + (a2 - alpha2)*y2*self.K[i2,i] - new_b + self.b
        
        self.b = new_b
        self.alphas[i1] = a1
        self.alphas[i2] = a2
                
        for i in [i1, i2]:
            a = self.alphas[i]
            if a > self.epsilon and a < self.C - self.epsilon and i not in self.non_bound_idx:
                self.non_bound_idx = np.hstack((self.non_bound_idx, i))

                                        
        return 1