import numpy as np
from scipy import sparse;
import sys
sys.path.append("..")

import sptensor
import ktensor
import CP_APR
import tensor;
import tenmat;
import accumarray;
import khatrirao;
import time;


def __calculatePi(X, M, R, n, N):
    """
    Calculate the product of all matrices but the n-th (Eq 3.6 in Chi + Kolda ArXiv paper)
    """    
    Pi = None
    if X.__class__ == sptensor.sptensor:
        Pi = np.ones((X.nnz(), R))
        for nn in np.concatenate((np.arange(0, n), np.arange(n+1, N))):
            Pi = np.multiply(M.U[nn][X.subs[:, nn],:], Pi)
    else:
        Pi = khatrirao.khatrirao_array([M.U[i] for i in range(len(M.U)) if i != n], reverse=True)
    return Pi

def __calculatePhi(X, M, R, n, Pi, epsilon):
    """
    Calculate the matrix for multiplicative update
    """
    Phi = None
    if X.__class__ == sptensor.sptensor:
        Phi = -np.ones((X.shape[n], R))
        xsubs = X.subs[:,n]
        v = np.sum(np.multiply(M.U[n][xsubs,:], Pi), axis=1)
        wvals = X.vals.flatten() / np.maximum(v, epsilon)
        for r in range(R):
            #Phi[:,r] = tools.accum_np(xsubs, np.multiply(wvals, Pi[:,r]), X.shape[n])
            Phi[:,r] = accumarray.accum_np(xsubs, np.multiply(wvals, Pi[:,r]), size=X.shape[n])
    else:
        Xn = tenmat.tenmat(X,[n])
        V = np.inner(M.U[n],Pi)
        W = Xn.data / np.maximum(V, epsilon)
        Phi = np.inner(W, Pi.transpose())
        
    return Phi

def __solveSubproblem(X, M, R, n, N, maxInner, epsilon, tol):
    """ """
    isConverged = True
    # Shift the weight from lambda to mode n 
    # B = A(n)*Lambda
    M.redistribute(n)
    # Pi(n) = [A(N) kr A(N-1) kr ... A(n+1) kr A(n-1) kr .. A(1)]^T
    Pi = __calculatePi(X, M, R, n, N) # Calculate product of all matrices but the nth one
    for i in range(maxInner):
        # Phi = (X(n) elem-div (B Pi)) Pi^T
        Phi = __calculatePhi(X, M, R, n, Pi, epsilon)
        # check for convergence that min(B(n), E - Phi(n)) = 0 [or close]
        kktModeViolations = np.max(np.abs(np.minimum(M.U[n], 1-Phi).flatten()))
        if (kktModeViolations < tol):
            break
        else:
            isConverged = False
            
        # Do the multiplicative update
        M.U[n] = np.multiply(M.U[n],Phi)
        print(" Mode={0}, Inner Iter={1}, KKT violation={2}".format(n, i, kktModeViolations))
    # Shift weight from mode n back to lambda
    M.normalize_mode(n,1)
    return M, Phi, i, kktModeViolations, isConverged

def __lsqr_fit(X, M):
    """ Calculate the fraction of the residual explained by the factorization """
    normX = X.norm();
    normresidual = np.sqrt(np.square(normX) + np.square(M.norm()) - 2*M.innerprod(X));
    fit = 1 - (normresidual / normX);
    return fit

def __loglikelihood(X,MF):
    """ 
    Computes the log-likelihood of model M given data X.
    Specifically, ll = -(sum_i m_i - x_i * log_i) where i is a
    multiindex across all tensor dimensions
    
    Parameters
    ----------
    X - input tensor of the class tensor or sptensor
    MF - ktensor

    Returns
    -------
    out : log likelihood value
    """
    N = X.ndims();
    # make a copy of the tensor so absorbing won't affect it
    M = ktensor.copyTensor(MF)    
    M.normalize_absorb(0, 1);
    ll = 0;
    
    if X.__class__ == sptensor.sptensor:
        xsubs = X.subs;
        A = M.U[0][xsubs[:,0], :];
        for n in range(1, N):
            A = np.multiply(A, M.U[n][xsubs[:,n],:]);
        ll = np.sum(np.multiply(X.vals.flatten(), np.log(np.sum(A, axis=1)))) - np.sum(M.U[0]);
    else:
        ## fill in what to do when it's not sparse tensor
        ll = -np.sum(M.U[0]);
    return ll;

def cp_apr(X, R, Minit, outputfile, tol=1e-4, maxiters=150, 
           maxinner=10, epsilon=1e-10, kappatol=1e-10, kappa=1e-2):
    N = X.ndims()
    nInnerIters = np.zeros(maxiters);
    
    ## Initialize M and Phi for iterations
    M = Minit
    prevM = ktensor.copyTensor(M)
    M.normalize(1)
    Phi = [[] for i in range(N)]
    kktModeViolations = np.zeros(N)
    kktViolations = -np.ones(maxiters)
    nViolations = np.zeros(maxiters)
    
    ## statistics
    cpStats = np.zeros(7)
    fmsStats = np.zeros(3)
    
    for iter in range(maxiters):
        startIter = time.time()
        isConverged = True;
        for n in range(N):
            startMode = time.time()
            ## Make adjustments to M[n] entries that violate complementary slackness
            if iter > 0:
                V = np.logical_and(Phi[n] > 1, M.U[n] < kappatol)
                if np.count_nonzero(V) > 0:
                    nViolations[iter] = nViolations[iter] + 1
                    M.U[n][V > 0] = M.U[n][V > 0] + kappa
            # solve the inner problem
            M, Phi[n], inner, kktModeViolations[n], isConverged = __solveSubproblem(X, M, R, n, N, maxinner, epsilon, tol)
            nInnerIters[iter] = nInnerIters[iter]+(inner+1)
            elapsed = time.time()-startMode
            # only write the outer iterations for now
            cpStats = np.vstack((cpStats, np.array([iter, n, inner, __lsqr_fit(X,M), __loglikelihood(X,M), kktModeViolations[n], elapsed])))
        kktViolations[iter] = np.max(kktModeViolations);
        elapsed = time.time()-startIter
        #cpStats = np.vstack((cpStats, np.array([iter, -1, -1, kktViolations[iter], __loglikelihood(X,M), elapsed])))
        print("Iteration {0}: Inner Its={1} with KKT violation={2}, nViolations={3}, and elapsed time={4}".format(iter, nInnerIters[iter], kktViolations[iter], nViolations[iter], elapsed));
        __writeDBFile(M, outputfile.format(iter), iter)
        fmsStats = np.vstack((fmsStats, np.array([iter, M.top_fms(prevM), M.greedy_fms(prevM)])))
        prevM = ktensor.copyTensor(M)
        
        if isConverged:
            break;
        
    cpStats = np.delete(cpStats, (0), axis=0) # delete the first row
    fmsStats = np.delete(fmsStats, (0), axis=0) # delete the first row
    
    ## print out the statistics
    fit = __lsqr_fit(X,M)
    ll = __loglikelihood(X,M)
    print("Number of iterations = {0}".format(iter))
    print("Final least squares fit = {0}".format(fit));
    print("Final log-likelihood = {0}".format(ll));
    print("Final KKT Violation = {0}".format(kktViolations[iter]))
    print("Total inner iterations = {0}".format(np.sum(nInnerIters)));
    modelStats = {"Iters" : iter, "LS" : fit, "LL" : ll, "KKT" : kktViolations[iter]}
    return M, cpStats, fmsStats, modelStats;

def __writeDBFile(X, filename, iter):
    # for each lambda create the stack
    idx = np.flatnonzero(X.lmbda)
    tempOut = np.column_stack((np.repeat(-1, len(idx)), np.repeat(-1, len(idx)), idx, X.lmbda[idx]))
    for n in range(X.ndims()):
        for r in range(X.R):
            idx = np.flatnonzero(X.U[n][:, r])
            # get the ones for this mode/factor
            temp = np.column_stack((np.repeat(n, len(idx)), idx, np.repeat(r, len(idx)), X.U[n][idx, r]))
            tempOut = np.vstack((tempOut, temp))
    tempOut = np.column_stack((np.repeat(modelID, tempOut.shape[0]), np.repeat(iter, tempOut.shape[0]), tempOut))
    np.savetxt(filename, tempOut, fmt="%s", delimiter="|")

iter = 200
R = 100

modelID = 5
labelID = 1
outfile = 'results/iter-db-5-{0}.csv'
set_desc = 'HF Patients Level 0 seed 0'
infile = file("data/hf-tensor-label1-level0-data.dat", "rb")

sqlLoadFile = "results/iter-{0}.sql".format(modelID)
statsFile = "results/iter-stats-{0}.csv".format(modelID)
fmsFile = "results/iter-fms-{0}.csv".format(modelID)

# load the sparse tensor information
subs = np.load(infile)
vals = np.load(infile)
siz = np.load(infile)
infile.close()
# now factor it
X = sptensor.sptensor(subs, vals, siz)
# Create a random initialization
N = X.ndims()
np.random.seed(0)
F = [];
for n in range(N):
    F.append(np.random.rand(X.shape[n], R))

Minit = ktensor.ktensor(np.ones(R), F)
Y, ystats, fmsStats, mstats = cp_apr(X, R, Minit=Minit, outputfile=outfile, maxiters=iter)

## automate the creation of the sql file
ystats = np.column_stack((np.repeat(modelID, ystats.shape[0]), ystats))
np.savetxt(statsFile, ystats, delimiter="|")

fmsStats = np.column_stack((np.repeat(modelID, fmsStats.shape[0]), fmsStats))
np.savetxt(fmsFile, fmsStats, delimiter="|")

sqlLoad = file(sqlLoadFile, "w")
for i in range(iter):
    dbFile = outfile.format(i)
    sqlLoad.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_iter_factors;\n".format(dbFile))

sqlLoad.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_iter_results;\n".format(statsFile))
sqlLoad.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_iter_fms;\n".format(fmsFile))
sqlLoad.write("insert into joyceho.tensor_iter_models values({0}, {1}, \'{2}\', {3}, {4}, {5}, {6}, {7});\n".format(modelID, labelID, set_desc, iter, 10, mstats['LS'], mstats['LL'], mstats['KKT']))

sqlLoad.close()