""" 
File that contains common methods for tensor methods
"""
import numpy as np
from sklearn.preprocessing import normalize

import accumarray
import sptensor

def randomInit(modeDim, R):
    """
    Randomly initialize column normalized matrices
    """
    U = []
    for mode in modeDim:
        tmp = np.random.rand(mode, R)
        # normalize it with the 1 norm
        norm_tmp = normalize(tmp, axis=0, norm='l1')
        U.append(norm_tmp)
    return U

def uniformInit(modeDim, R):
    """
    Initialize the matrix using a uniform distribution
    """
    U = []
    for mode in modeDim:
        tmp = np.ones((mode, R))
        # normalize it with the 1 norm
        norm_tmp = normalize(tmp, axis=0, norm='l1')
        U.append(norm_tmp)
    return U

def countNonZeros(U):
	"""
	Count the number of non-zero elements for each column
	"""
	return [np.count_nonzero(U[:, r]) for r in range(U.shape[1])]

def countTensorNNZ(M):
	"""
	Count the number of non-zero elements for each mode
	"""
	return [countNonZeros(M.U[mode]) for mode in range(M.ndims())]

def hardThresholdMatrix(U, thresh):
	""" 
	Perform hard thresholding of a matrix
    
    Parameters
    ------------
    U : the matrix to threshold
    thresh : the threshold value (anything below is chopped to zero)
    
    Output
    -----------
    U: the new thresholded matrix
    """
	zeroIdx = np.where(U < thresh)
	U[zeroIdx] = 0
	return U

def hardThresholdFactors(M, modes, thresh=1e-2):
	""" 
	Perform hard thresholding of a set of factors
    
    Parameters
    ------------
    M : a list of matrices to threshold
    modes : a list of (tensor index, mode index) to threshold
    
    Output
    -----------
    U: the new thresholded matrix
    """
	for m in range(len(modes)):
		tensorIdx = modes[m][0]
        tensorDim = modes[m][1]
        # print tensorIdx, tensorDim
        M[tensorIdx].U[tensorDim] = hardThresholdMatrix(M[tensorIdx].U[tensorDim], thresh)
	return M

def calculatePi(X, M, n):
    """
    Calculate the product of all matrices but the n-th (Eq 3.6 in Chi + Kolda ArXiv paper)
    """    
    Pi = None
    if X.__class__ == sptensor.sptensor:
        Pi = np.ones((X.nnz(), M.R))
        for nn in np.concatenate((np.arange(0, n), np.arange(n+1, X.ndims()))):
            Pi = np.multiply(M.U[nn][X.subs[:, nn],:], Pi)
    else:
        Pi = khatrirao.khatrirao_array([M.U[i] for i in range(len(M.U)) if i != n], reverse=True)
    return Pi

def calculatePhi(X, B, Pi, n, epsilon=1e-4, C=None):
    """
    Calculate the matrix for multiplicative update

    Parameters
    ----------
    X       : the observed tensor
    B       : the factor matrix associated with mode n
    Pi      : the product of all matrices but the n-th from above
    n       : the mode that we are trying to solve the subproblem for
    epsilon : the 
    C       : the augmented / non-augmented tensor (\alpha u \Psi or B \Phi) in sparse form
    """
    Phi = None
    if X.__class__ == sptensor.sptensor:
        Phi = -np.ones((X.shape[n], B.shape[1]))
        xsubs = X.subs[:,n]
        if C !=  None:
            v = np.sum(np.multiply(B[xsubs,:], Pi) + C, axis=1)
        else:
            v = np.sum(np.multiply(B[xsubs,:], Pi), axis=1)
        wvals = X.vals.flatten() / v
        for r in range(B.shape[1]):
            Phi[:,r] = accumarray.accum_np(xsubs, np.multiply(wvals, Pi[:,r]), size=X.shape[n])
    else:
        Xn = tenmat.tenmat(X,[n])
        V = np.inner(B,Pi)
        W = Xn.data / np.maximum(V, epsilon)
        Phi = np.inner(W, Pi.transpose())  
    return Phi

def lsqrFit(X, M):
    """ 
    Calculate the fraction of the residual explained by the factorization 
    Parameters
    ------------
    X : observed tensor
    M : factorized tensor
    """
    normX = X.norm();
    normresidual = np.sqrt(np.square(normX) + np.square(M.norm()) - 2*M.innerprod(X));
    fit = 1 - (normresidual / normX);
    return fit

def loglikelihood(X, M):
    """ 
    Computes the log-likelihood of model M given data X.
    Specifically, ll = -(sum_i m_i - x_i * log_i) where i is a
    multiindex across all tensor dimensions
    
    Parameters
    ----------
    X : input tensor of the class tensor or sptensor
    M : ktensor

    Returns
    -------
    out : log likelihood value
    """
    N = X.ndims();
    ll = 0;

    if X.__class__ == sptensor.sptensor:
        xsubs = X.subs;
        C = []
        for m in range(len(M)):
            C.append(np.dot(M[m].U[0], np.diag(M[m].lmbda))[xsubs[:,0], :])
        for n in range(1, N):
            for m in range(len(M)):
                C[m] = np.multiply(C[m], M[m].U[n][xsubs[:,n], :])
        MHat = np.zeros(C[0].shape)
        for m in range(len(M)):
            MHat = MHat + C[m]
        ll = np.sum(np.multiply(X.vals.flatten(), np.log(np.sum(MHat, axis=1)))) - np.sum(np.sum(MHat, axis=1));
    else:
        ## fill in what to do when it's not sparse tensor
        return ll
    return ll