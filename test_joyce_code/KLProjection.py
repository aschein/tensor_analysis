'''
Compute the projection onto a mode using the specified factorization as the basis 
'''
import numpy as np
import ktensor
import CP_APR

class KLProjection:
    basis = None
    R = 0
    
    def __init__(self, U, R):
        self.basis = U
        self.R = R
        ## Verify the shape of the values are correct
        for n in range(len(U)):
            if U[n].shape[1] != self.R:
                raise ValueError("Shape of the basis factors does not match the rank R")

    def projectSlice(self, X, n, iters=10, epsilon=1e-10, convTol=1e-4):
        """ 
        Project a slice, solving for the factors of the nth mode
        
        Parameters
        ------------
        X : the tensor to project onto the basis
        n : the mode to project onto
        iters : the max number of inner iterations
        epsilon : parameter to avoid dividing by zero
        convTol : the convergence tolerance
        
        Output
        -----------
        the projection matrix
        """
        ## Setup the 'initial guess'
        F = []
        for m in range(X.ndims()):
            if m == n:
                F.append(np.random.rand(X.shape[m], self.R));
            else:
                ## double check the shape is the right dimensions
                if (self.basis[m].shape[0] != X.shape[m]):
                    raise ValueError("Shape of the tensor X is incorrect");
                F.append(self.basis[m])
        M = ktensor.ktensor(np.ones(self.R), F);
        ## Solve for the subproblem
        M, Phi, totIter, kktMV = CP_APR.solveForModeB(X, M, n, iters, epsilon, convTol)
        ## scale by summing across the rows
        totWeight = np.sum(M.U[n], axis=1)
        zeroIdx = np.where(totWeight < 1e-100)[0]
        if len(zeroIdx) > 0:
            # for the zero ones we're going to evenly distribute
            evenDist = np.repeat(1.0 / self.R, len(zeroIdx)*self.R)
            M.U[n][zeroIdx, :] = evenDist.reshape((len(zeroIdx), self.R))
            totWeight = np.sum(M.U[n], axis=1)
        twMat = np.repeat(totWeight, self.R).reshape(X.shape[n], self.R)
        M.U[n] = M.U[n] / twMat
        return M.U[n]