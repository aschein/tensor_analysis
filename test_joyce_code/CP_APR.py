'''
Compute the nonnegative tensor factorization using alternating Poisson regression
'''
import numpy as np
import time

import ktensor
import sptensor
import tensorTools

def solveForModeB(X, M, n, maxInner, epsilon, tol):
    """ 
    Solve for the subproblem B = argmin (B >= 0) f(M)
    
    Parameters
    ----------
    X : the original tensor
    M : the current CP factorization
    n : the mode that we are trying to solve the subproblem for
    epsilon : parameter to avoid dividing by zero
    tol : the convergence tolerance (to 

    Returns
    -------
    M : the updated CP factorization
    Phi : the last Phi(n) value 
    iter : the number of iterations before convergence / maximum
    kktModeViolation : the maximum value of min(B(n), E - Phi(n))
    """
    # Pi(n) = [A(N) kr A(N-1) kr ... A(n+1) kr A(n-1) kr .. A(1)]^T
    Pi = tensorTools.calculatePi(X, M, n)
    for iter in range(maxInner):
        # Phi = (X(n) elem-div (B Pi)) Pi^T
        Phi = tensorTools.calculatePhi(X, M.U[n], Pi, n, epsilon=epsilon)
        # check for convergence that min(B(n), E - Phi(n)) = 0 [or close]
        kktModeViolation = np.max(np.abs(np.minimum(M.U[n], 1-Phi).flatten()))
        if (kktModeViolation < tol):
            break
        # Do the multiplicative update
        M.U[n] = np.multiply(M.U[n],Phi)
        #print(" Mode={0}, Inner Iter={1}, KKT violation={2}".format(n, iter, kktModeViolation))
    return M, Phi, iter, kktModeViolation

def __solveSubproblem(X, M, n, maxInner, isConverged, epsilon, tol):
    """ """
    # Shift the weight from lambda to mode n 
    # B = A(n)*Lambda
    M.redistribute(n)
    # solve the inner problem
    M, Phi, iter, kktModeViolation = solveForModeB(X, M, n, maxInner, epsilon, tol)
    if (iter > 0):
        isConverged = False
    # Shift weight from mode n back to lambda
    M.normalize_mode(n,1)
    return M, Phi, iter, kktModeViolation, isConverged
    
def cp_apr(X, R, Minit=None, tol=1e-4, maxiters=1000, maxinner=10, 
           epsilon=1e-10, kappatol=1e-10, kappa=1e-2):
    """ 
    Compute nonnegative CP with alternative Poisson regression.
    Code is the python implementation of cp_apr in the MATLAB Tensor Toolbox 
    
    Parameters
    ----------
    X : input tensor of the class tensor or sptensor
    R : the rank of the CP
    Minit : the initial guess (in the form of a ktensor), if None random guess
    tol : tolerance on the inner KKT violation
    maxiters : maximum number of iterations
    maxinner : maximum number of inner iterations
    epsilon : parameter to avoid dividing by zero
    kappatol : tolerance on complementary slackness
    kappa : offset to fix complementary slackness

    Returns
    -------
    M : the CP model as a ktensor
    cpStats: the statistics for each inner iteration
    modelStats: a dictionary item with the final statistics for this tensor factorization
    """
    N = X.ndims()
     
    ## Random initialization
    if Minit == None:
        F = tensorTools.randomInit(X.shape, R)
        Minit = ktensor.ktensor(np.ones(R), F);
        
    nInnerIters = np.zeros(maxiters);
    
    ## Initialize M and Phi for iterations
    M = Minit
    M.normalize(1)
    Phi = [[] for i in range(N)]
    kktModeViolations = np.zeros(N)
    kktViolations = -np.ones(maxiters)
    nViolations = np.zeros(maxiters)
    
    ## statistics
    cpStats = np.zeros(7)
    
    for iteration in range(maxiters):
        startIter = time.time()
        isConverged = True;
        for n in range(N):
            startMode = time.time()
            ## Make adjustments to M[n] entries that violate complementary slackness
            if iteration > 0:
                V = np.logical_and(Phi[n] > 1, M.U[n] < kappatol)
                if np.count_nonzero(V) > 0:
                    nViolations[iteration] = nViolations[iteration] + 1
                    M.U[n][V > 0] = M.U[n][V > 0] + kappa
            M, Phi[n], inner, kktModeViolations[n], isConverged = __solveSubproblem(X, M, n, maxinner, isConverged, epsilon, tol)
            elapsed = time.time() - startMode
            # only write the outer iterations for now
            cpStats = np.vstack((cpStats, np.array([iteration, n, inner, tensorTools.lsqrFit(X,M), tensorTools.loglikelihood(X,[M]), kktModeViolations[n], elapsed])))

        kktViolations[iteration] = np.max(kktModeViolations);
        elapsed = time.time()-startIter
        #cpStats = np.vstack((cpStats, np.array([iter, -1, -1, kktViolations[iter], __loglikelihood(X,M), elapsed])))
        print("Iteration {0}: Inner Its={1} with KKT violation={2}, nViolations={3}, and elapsed time={4}".format(iteration, nInnerIters[iteration], kktViolations[iteration], nViolations[iteration], elapsed));
        if isConverged:
            break;
    
    cpStats = np.delete(cpStats, (0), axis=0) # delete the first row which was superfluous
    ### Print the statistics
    fit = tensorTools.lsqrFit(X,M)
    ll = tensorTools.loglikelihood(X,[M])
    print("Number of iterations = {0}".format(iteration))
    print("Final least squares fit = {0}".format(fit))
    print("Final log-likelihood = {0}".format(ll))
    print("Final KKT Violation = {0}".format(kktViolations[iteration]))
    print("Total inner iterations = {0}".format(np.sum(nInnerIters)))
    
    modelStats = {"Iters" : iter, "LS" : fit, "LL" : ll, "KKT" : kktViolations[iteration]}
    return M, cpStats, modelStats