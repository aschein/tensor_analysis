'''
Compute the projection onto a mode using the specified factorization.
This is the multi-tensor implementation and generalization of KLProjection
'''
import numpy as np
import ktensor
import sim_APR

class simKLProjection:
    basis = None
    R = 0
    
    def __init__(self, M, R):
        ## make a copy of the original basis
        self.basis = [ktensor.copyTensor(m) for m in M]
        self.R = R

    def projectSlice(self, X, sm, iters=10, epsilon=1e-10, convTol=1e-4):
        """ 
        Project the mode onto the basis
        
        Parameters
        ------------
        X : the tensor list to project
        sm : the mode to project onto, a 2d array specifying the tensor mode locations
        iters : the max number of inner iterations
        epsilon : parameter to avoid dividing by zero
        convTol : the convergence tolerance
        
        Output
        -----------
        the projection matrix
        """
        ## Setup the 'initial guess' for the shared mode
        firstI = sm[0,0]    # first i that should be updated
        firstN = sm[0,1]    # first n that should be updated
        self.basis[firstI].U[firstN] = np.random.rand(X[firstI].shape[firstN], self.R) 
        self.basis = sim_APR.updateSharedFactors(self.basis, sm)
        for i in range(len(self.basis)):
            ## set the lambdas to 0
            self.basis[i].lmbda = np.ones(self.R)
        isConverged = True
        ## Solve for the subproblem
        M, PhiHat, totIter, kktMV, isConverged = sim_APR.solveSharedMode(X, self.basis, self.R, sm, isConverged, iters, epsilon, convTol)
        ## scale by summing across the rows
        totWeight = np.sum(M[firstI].U[firstN], axis=1)
        zeroIdx = np.where(totWeight < 1e-100)[0]
        if len(zeroIdx) > 0:
            # for the zero ones we're going to evenly distribute
            evenDist = np.repeat(1.0 / self.R, len(zeroIdx)*self.R)
            M[firstI].U[firstN][zeroIdx, :] = evenDist.reshape((len(zeroIdx), self.R))
            totWeight = np.sum(M[firstI].U[firstN], axis=1)
        twMat = np.repeat(totWeight, self.R).reshape(X[firstI].shape[firstN], self.R)
        M[firstI].U[firstN] = M[firstI].U[firstN] / twMat
        return M[firstI].U[firstN]