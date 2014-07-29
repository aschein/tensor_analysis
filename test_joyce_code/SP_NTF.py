'''
Compute the nonnegative tensor factorization using alternating Poisson regression

This is the algorithm described in the paper Marble.
'''
import numpy as np
import time
from collections import OrderedDict

import ktensor
import sptensor
import tensorTools

#### CLASS VARIABLES #####
AUG_LOCATION = 1
REG_LOCATION = 0
AUG_MIN = 1e-10

class SP_NTF:
    X = None        ## Observed tensors
    N = 0           ## The number of dimensions
    R = 0           ## Rank of decomposition
    M = {REG_LOCATION : None, AUG_LOCATION: None}   ## Decomposed tensor
    convTol = 0     ## Convergence tolerance of subproblem
    maxIters = 0    ## Maximum number of iterations before returning
    maxInnerIters = 0   ## Maximum number of iterations per mode
    dlTol = 0       ## Convergence difference in KL divergence
    alpha = 0

    def __init__(self, X, R, alpha=1, dlTol = 1e-2, tol=1e-4, maxiters=100, maxinner=5):
        self.X = X
        self.N = X.ndims()
        self.R = R
        self.alpha = alpha
        self.dlTol = dlTol
        self.convTol = tol
        self.maxIters = maxiters
        self.maxInnerIters = maxinner

    def initialize(self, M=None):
        """
        Initialize the tensor decomposition
        """
        if M == None:
            AU = tensorTools.randomInit(self.X.shape, 1)
            F = tensorTools.randomInit(self.X.shape, self.R)
            self.M[REG_LOCATION] = ktensor.ktensor(np.ones(self.R), F)
            self.M[AUG_LOCATION] = ktensor.ktensor(np.ones(1), AU)
        else:
            ## do a quick sanity check
            if len(M) != 2:
                raise ValueError("Initialization needs to be of size 2")
            if M[0].__class__ != ktensor.ktensor and M[1].__class__ != ktensor.ktensor:
                raise ValueError("Not ktensor type")
            self.M = M

    def normalizeAugTensor(self):
        """
        Normalize the augmented tensor to the value alpha
        """
        self.M[AUG_LOCATION].normalize(1)
        self.M[AUG_LOCATION].lmbda = np.repeat(self.alpha, 1)

    def __solveMode(self, Pi, B, C, n):
        """
        Performs the inner iterations and checks for convergence.
        """
        for innerI in range(self.maxInnerIters):
            # Phi = (X(n) elem-div (B Pi)) Pi^T
            Phi = tensorTools.calculatePhi(self.X, B, Pi, n, C=C)
            # check for convergence that min(B(n), E - Phi(n)) = 0 [or close]
            kktModeViolation = np.max(np.abs(np.minimum(B, 1-Phi).flatten()))
            if (kktModeViolation < self.convTol):
                break
            # Do the multiplicative update
            B = np.multiply(B, Phi)
        return B, (innerI+1), kktModeViolation

    def solveSubproblem(self, C, aug, n):
        """
        Solve the subproblem for mode n

        Parameters
        ------------
        C : the "other" tensor, either bias or signal of the tensor decomposition
        aug : the location to solve (either augmented or not)
        n : the mode to solve
        """
        ## shift the weight from lambda to mode n
        self.M[aug].redistribute(n)
        Pi = tensorTools.calculatePi(self.X, self.M[aug], n)
        B, inI, kktModeViolation = self.__solveMode(Pi, self.M[aug].U[n], C, n)
        self.M[aug].U[n] = B
        ## shift the weight from mode to lambda
        self.M[aug].normalize_mode(n, 1)
        return B, Pi, inI, kktModeViolation

    def __solveAugmentedTensor(self, xsubs, B, Pi, n):
        # now that we are done, we can calculate the new 'unaugmented matricization'
        Chat = np.multiply(B[xsubs, :], Pi)
        B, Pi, inI2, kktModeViolation2 = self.solveSubproblem(Chat, AUG_LOCATION, n)
        self.M[AUG_LOCATION].U[n] = np.maximum(np.tile(AUG_MIN, self.M[AUG_LOCATION].U[n].shape), self.M[AUG_LOCATION].U[n])
        self.normalizeAugTensor()
        return inI2, kktModeViolation2

    def __solveSignalTensor(self, xsubs, BHat, n):
        ## calculate Psi
        Psi = tensorTools.calculatePi(self.X, self.M[AUG_LOCATION], n)
        C = np.multiply(BHat[xsubs,:], Psi)
        return self.solveSubproblem(C, REG_LOCATION, n)

    def computeDecomp(self, gamma=None, gradual=True):
        ## random initialize if not existing
        if self.M[REG_LOCATION] == None and self.M[AUG_LOCATION] == None:
            self.initialize()
        ## Kkeep track of the iteration information
        iterInfo = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
        lastLL = tensorTools.loglikelihood(self.X, self.M)
        ## projection factor starts at 0 (unless there's no gradual)
        xi = 0 if gradual else 1
        ## if nothing is set, we're just not going to do any hard-thresholding
        if gamma == None:
            gamma = list(np.repeat(0, self.N))
        ## for outer iterations
        for iteration in range(self.maxIters):
            startIter = time.time()
            for n in range(self.N):
                startMode = time.time()
                ## first we calculate the "augmented" tensor matricization
                self.M[AUG_LOCATION].redistribute(n)
                xsubs = self.X.subs[:,n]
                B, Pi, inI1, kktModeViolation1 = self.__solveSignalTensor(xsubs, self.M[AUG_LOCATION].U[n], n)
                ## hard threshold based on the xi and gamma
                thr = xi * gamma[n]
                if (thr > 0):
                    self.M[REG_LOCATION].U[n] = tensorTools.hardThresholdMatrix(self.M[REG_LOCATION].U[n], thr)
                    # renormalize the mode
                    self.M[REG_LOCATION].normalize_mode(n, 1)
                    ## recalculate B using the new matrix
                    B = np.dot(self.M[REG_LOCATION].U[n], np.diag(self.M[REG_LOCATION].lmbda))
                elapsed1 = time.time() - startMode
                # now that we are done, we can calculate the new 'unaugmented matricization'
                inI2, kktModeViolation2 = self.__solveAugmentedTensor(xsubs, B, Pi, n)
                elapsed2 = time.time() - startMode
                ll = tensorTools.loglikelihood(self.X, self.M)
                iterInfo[str((iteration, n))] = { "Time": [elapsed1, elapsed2], 
                    "KKTViolation": [kktModeViolation1, kktModeViolation2],
                    "Iterations": [inI1, inI2],
                    "LL": ll}
            if gradual:
                xiTemp = 1-np.min([1, (np.absolute(lastLL - ll) / np.max(np.absolute([lastLL,ll])))])
                if xiTemp > xi:
                    ## take the mean of the two
                    xi = (xi + xiTemp) / 2
            print("Iteration {0}: Xi = {1}, dll = {2}, time = {3}".format(iteration, xi, np.abs(lastLL - ll), time.time() - startIter))
            if np.abs(lastLL - ll) < self.dlTol and xi >= 0.99:
                break;
            lastLL = ll
        return iterInfo

    def projectData(self, XHat, n, maxiters=10, maxinner=10):
        ## store off the old ones
        origM = {REG_LOCATION: ktensor.copyTensor(self.M[REG_LOCATION]), AUG_LOCATION: ktensor.copyTensor(self.M[AUG_LOCATION])} 
        origX = self.X
        self.X = XHat
        ## randomize the nth 
        self.M[REG_LOCATION].U[n] = np.random.rand(self.X.shape[n], self.R)
        self.M[REG_LOCATION].lmbda = np.ones(self.R)
        self.M[AUG_LOCATION].U[n] = np.random.rand(self.X.shape[n], 1)
        self.M[AUG_LOCATION].lmbda = np.ones(1)
        ## renormalize
        self.M[REG_LOCATION].normalize(1)
        self.normalizeAugTensor()
        lastLL = tensorTools.loglikelihood(self.X,self.M)
        for iteration in range(maxiters):
            xsubs = self.X.subs[:,n]
            B, Pi, inI1, kktModeViolation1 = self.__solveSignalTensor(xsubs, self.M[AUG_LOCATION].U[n], n)
            inI2, kktModeViolation2 = self.__solveAugmentedTensor(xsubs, B, Pi, n)
            ll = tensorTools.loglikelihood(self.X,self.M)
            if np.abs(lastLL - ll) < self.dlTol:
                break
            lastLL = ll
        ## scale by summing across the rows
        totWeight = np.sum(self.M[REG_LOCATION].U[n], axis=1)
        zeroIdx = np.where(totWeight < 1e-100)[0]
        if len(zeroIdx) > 0:
            evenDist = 1.0 / self.M[REG_LOCATION].R
            self.M[REG_LOCATION].U[n][zeroIdx, :] = np.tile(evenDist, (len(zeroIdx), self.M[REG_LOCATION].R))
            totWeight = np.sum(self.M[REG_LOCATION].U[n], axis=1)
        twMat = np.repeat(totWeight, self.M[REG_LOCATION].R).reshape(self.X.shape[n], self.M[REG_LOCATION].R)
        projMat = self.M[REG_LOCATION].U[n] / twMat
        biasMat = self.M[AUG_LOCATION].U[n]
        self.M = origM
        self.X = origX
        return projMat, biasMat