import numpy as np
import shelve
import sptensor
import CP_APR

def loadAxisInfo(filename):
    tensorInfo = shelve.open(filename, "r")
    axisList = tensorInfo['axis']
    tensorInfo.close()
    return axisList

def decomposeCountTensor(filename, R, outerIters=20, innerIters=10, convergeTol=1e-2, zeroTol=1e-4):
    """
    Given a file, load the tensor data and then 
    From a file, load the tensor data and 
    then decompose using CP_APR with specified rank
    
    Parameters:
    filename - the file that stores the sparse tensor representation using numpy
    R - the rank of the tensor
    outerIters - the maximum number of outer iterations
    innerIters - the maximum number of inner iterations
    convergeTol - the convergence tolerance
    zeroTol - the amount to zero out the factors
    
    Output:
    
    """
    X = sptensor.loadTensor(filename)
    Y, iterStats, modelStats = CP_APR.cp_apr(X, R, tol=convergeTol, maxiters=outerIters, maxinner=innerIters)
    # normalize the factors using the 1 norm and then sort in descending order
    Y.normalize_sort(1)
    Y = zeroSmallFactors(Y, zeroThr=zeroTol)
    return Y, iterStats, modelStats

def zeroSmallFactors(X, zeroThr=1e-4):
    """ Convert the small factors to 0"""
    for n in range(X.ndims()):
        zeroIdx = np.where(X.U[n] < zeroThr)
        X.U[n][zeroIdx] = 0
    return X

def getDBOutput(X, axisList=None):
    """ Get the raw DB output, which assumes it will be mode, feature_name, factor, value"""
    # create the index for the axis list
    if axisList == None:
        axisList = []
        for n in range(X.ndims()):
            axisList.append(range(X.shape[n]))
    # for each lambda create the stack
    idx = np.flatnonzero(X.lmbda)
    lmbda_feat = np.array(np.repeat("lambda", len(idx)), dtype="|S100")
    tempOut = np.column_stack((np.repeat(-1, len(idx)), lmbda_feat[idx], idx, X.lmbda[idx]))
    for n in range(X.ndims()):
        for r in range(X.R):
            idx = np.flatnonzero(X.U[n][:, r])
            mode_feat = np.array(axisList[n], dtype="|S100")
            # get the ones for this mode/factor
            temp = np.column_stack((np.repeat(n, len(idx)), mode_feat[idx], np.repeat(r, len(idx)), X.U[n][idx, r]))
            tempOut = np.vstack((tempOut, temp))
    return tempOut
