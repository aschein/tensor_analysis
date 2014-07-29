'''
Compute nonnegative CP with alternative Poisson regression.
'''
import numpy as np
import time

import ktensor
import CP_APR

class SAPR:
    """ Simultaneous nonnegative multi-tensor factorizations """
    X = None
    R = 0
    modes = None
    nModes = 0
    M = None
    tol = 1e-4
    maxIters = 50
    maxInnerIters = 10
    epsilon = 1e-10
    kappaTol = 1e-10
    kappa = 1e-2
    
    @staticmethod
    def __randomInitialization(shape, R):
        F = []
        for n in range(len(shape)):
            F.append(np.random.rand(shape[n], R))
        return(ktensor.ktensor(np.ones(R), F))
    
    def __init__(self, X, R, sharedModes, Minit=None, tol=1e-4, maxiters=1000, maxinner=10, 
           epsilon=1e-10, kappatol=1e-10, kappa=1e-2):
        self.X = X
        self.R = R
        I = len(X)
        allModes = [range(X[i].ndims()) for i in range(I)]
        # calculate all the nodes that will need processing
        self.modes = list(sharedModes)
        for sm in sharedModes:
            for k in range(sm.shape[0]):
                ti = sm[k,0]    # index of tensor in X
                tm = sm[k,1]    # tensor mode
                if tm in allModes[ti]:
                    allModes[ti].remove(tm)
        for i in range(I):
            for k in range(len(allModes[i])):
                self.modes.append(np.array([i, allModes[i][k]]))
        self.nModes = len(self.modes)
        ## initialize M if necessary
        self.M = Minit
        if self.M == None:
            self.M = [SAPR.__randomInitialization(X[i].shape, R) for i in range(I)]  
        for i in range(I):
             self.M[i].normalize(1)
        ## update shared factors
        for sm in sharedModes:
            self.M = updateSharedFactors(self.M, sm)
        self.tol = tol
        self.maxIters = maxiters
        self.maxInnerIters = maxinner
        self.epsilon = epsilon
        self.kappaTol = kappatol
        self.kappa = kappa

    def shareLambda(self, i):
        """ 
        Set the lambda values to be shared across all the M[i]
        """
        temp = self.M[i].lmbda
        for k in range(len(self.M)):
            self.M[i].lmbda = temp
    
    def solveUnsharedMode(self, mode, isConverged):
        """ 
        Solve the unshared mode problem
        This is simply the same as the MM approach for CP-APR
        
        Parameters
        ----------
        mode : a length 2 array that contains the ith tensor in position 0 and the nth mode in position 1
        isConverged : passing along the convergence parameter
        """
        i = mode[0]
        n = mode[1]
        ## Shift the weight in factorization M(i) from lambda_i to mode n
        self.M[i].redistribute(n)
        self.M[i], Phi, iter, kttModeViolation = CP_APR.solveForModeB(self.X[i], self.M[i], n, self.maxInnerIters, self.epsilon, self.tol)
        if (iter > 0):
            isConverged = False
        # Shift weight from mode n back to lambda
        self.M[i].normalize_mode(n,1)
        ## Normalize the lambda to all the others
        self.shareLambda(i)
        return Phi, iter, kttModeViolation, isConverged

    def fixSlackness(self, Phi, PhiViolation, I, N):
        """ 
        Fix the complementary slackness to scootch values away from zero that shouldn't be zero
        """
        V = np.logical_and(Phi > PhiViolation, self.M[I].U[N] < self.kappaTol)
        if np.count_nonzero(V) > 0:
            self.M[I].U[N][V > 0] = self.M[I].U[N][V > 0] + self.kappa
    
    def logLikelihood(self):
        ## calculate the log likelihood for each tensor
        ll = [CP_APR.loglikelihood(self.X[i], self.M[i]) for i in range(len(self.X))]
        return np.sum(np.array(ll))
        
    def factorize(self):
        ## Get the Phi values (or Phi hat)
        Phi = [[] for i in range(self.nModes)]
        cpStats = np.zeros(6)
        
        ## Iterate through each mode
        for iter in range(self.maxIters):
            startIter = time.time()
            isConverged = True
            for m in range(self.nModes):
                startMode = time.time()
                mode = self.modes[m]
                print mode
                if mode.size > 2:
                    if iter > 0:
                        ## Sum the lambdas
                        Lmbda = np.zeros((1, self.R))
                        for k in range(mode.shape[0]):
                            Lmbda = Lmbda + self.M[mode[k,0]].lmbda
                        Lmbda = np.repeat(Lmbda, Phi[m].shape[0], axis=0)
                        ## update just the first one
                        self.fixSlackness(Phi[m], Lmbda, mode[0,0], mode[0,1])
                        self.M = updateSharedFactors(self.M, mode)
                    ## Solve the shared Mode
                    self.M, Phi[m], innerIter, kktModeViolation, isConverged = solveSharedMode(self.X, self.M, self.R, mode, isConverged, self.maxInnerIters, self.epsilon, self.tol)

                else:
                    if iter > 0:
                        self.fixSlackness(Phi[m], np.ones((Phi[m].shape)), mode[0], mode[1])
                    Phi[m], innerIter, kktModeViolation, isConverged = self.solveUnsharedMode(mode, isConverged)
                elapsed = time.time()-startMode
                cpStats = np.vstack((cpStats, np.array([iter, m, innerIter, self.logLikelihood(), kktModeViolation, elapsed])))
            elapsed = time.time()-startIter
            print("Iteration {0}: elapsed time={1}".format(iter, elapsed))
            if isConverged:
                break
        # delete the first statistics
        cpStats = np.delete(cpStats, (0), axis=0)
        modelStats = {"Iters": iter, "LS" : 0, "LL": self.logLikelihood(), "KKT": kktModeViolation}
        
        return self.M, cpStats, modelStats

def updateSharedFactors(M, sm):
    """ 
    Update all the other factor matrices that share this mode.
    The first row of the sm numpy array is the "true" factor that will be used.
    """
    row = sm.shape[0]           # get the number of modes that share this
    if row == 1:
        return M
    F = M[sm[0,0]].U[sm[0,1]]   # get the first factor
    for k in range(1, row):
        M[sm[k,0]].U[sm[k,1]] = F
    return M
    
def solveSharedMode(X, M, R, sm, isConverged, maxInnerIters=10, epsilon=1e-10, tol=1e-4):
    row = sm.shape[0]
    firstI = sm[0,0]    # first i that should be updated
    firstN = sm[0,1]    # first n that should be updated
    for iter in range(maxInnerIters):
        Phi = np.zeros(M[firstI].U[firstN].shape)
        for k in range(row):
            j = sm[k, 0]
            n = sm[k, 1]
            # calculate Pi
            Pi = CP_APR.calculatePi(X[j], M[j], n)
            Phi = Phi + CP_APR.calculatePhi(X[j], M[j], n, Pi, epsilon)
        # check for convergence
        kktModeViolation = np.max(np.abs(np.minimum(M[firstI].U[firstN], 1-PhiHat).flatten()))
        if (kktModeViolation < tol):
            break
        M[firstI].U[firstN] = np.multiply(M[firstI].U[firstN], Phi)
        M[firstI].normalize_mode(firstN, 1)
        # update the shared factors
        M = updateSharedFactors(M, sm)
    if (iter > 0):
        isConverged = False
    return M, PhiHat, iter, kktModeViolation, isConverged