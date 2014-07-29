import numpy as np
import random

import sys
sys.path.append("..")
import sptensor
import ktensor

def normalize(mat):
	## Normalize the matrix using the sum of each column
	normCol = np.sum(mat, axis=0)
	## if it's 0 set it to be 1
	normCol[np.where(normCol==0)] = 1
	normMat = np.tile(normCol, reps=mat.shape[0]).reshape(mat.shape)
	mat = np.divide(mat, normMat)
	return mat, normCol

def randomSampleIdx(size, fill, largePer=0.1, largeFact=20):
	tmp = np.zeros(size)
	# randomly select some entries to be nonzero
	nnz = random.sample(range(size), fill)
	# randomly sample numbers
	tmp[nnz] = np.random.random(fill)
	# percentage of large size
	totBig = min(int(largePer*fill), 1)
	bigIdx = random.sample(nnz, totBig)
	tmp[bigIdx] = largeFact * tmp[bigIdx]
	return tmp

def generateSolution(N, R, nSize, aFill, nFill):
	A = [np.zeros((nSize[n], R)) for n in range(N)]
	U = []
	# initialize lambda and the matrices
	lmbda = np.random.random_integers(low = 1, high = 1, size=R)
	for n in range(N):
		U.append(np.random.rand(nSize[n], 1))
		U[n], uNorm = normalize(U[n])
	for r in range(R):
		#the zeroth mode always has something
		A[0][:, r] = randomSampleIdx(nSize[0], aFill[0])
		## sample which should be non-zero modes for this R
		nnzMode = random.sample(range(1, N), nFill)
		## for each mode, we'll sample a set of indices
		for n in nnzMode:
			A[n][:, r] = randomSampleIdx(nSize[n], aFill[n])
	for n in range(N):
		## normalize the A matrices
		A[n], aNorm = normalize(A[n])
		lmbda = np.multiply(lmbda, aNorm)
	## reorder based on lambda
	sortidx = np.argsort(lmbda)[::-1]
	lmbda = lmbda[sortidx]
	for n in range(N):
		A[n] = A[n][:, sortidx]
	return lmbda, A, U

def generateOriginalTensor(L, A, U, tensorModes, alpha):
	MFull = []
	## for each set of modes, we will construct both M and MHat
	for k in range(len(tensorModes)):
		Alist = [A[n] for n in tensorModes[k]]
		Ulist = [U[n] for n in tensorModes[k]]
		M = ktensor.ktensor(L, Alist)
		Mhat = ktensor.ktensor(np.array([alpha]), Ulist)
		MFull.append(M.toTensor() + Mhat.toTensor())
	return MFull

def generateRandomProblem(MFull):
	X = []
	for M in MFull:
		## get the non-zero entries
		nnz = np.nonzero(M.data)
		mfVals = M.data.flatten()
		xVals = np.reshape([np.random.poisson(l) for l in mfVals], (len(mfVals), 1))
		xSubs = np.zeros((len(mfVals), M.ndims()))
		xSubs.dtype = 'int'
		for n in range(M.ndims()):
			xSubs[:, n] = nnz[n]
		## figure out which ones are non-zero and build X with it to avoid extraneous properties
		nnzX = np.nonzero(xVals)
		print "Number of nonzeros:" + str(len(nnzX[0]))
		xVals = xVals[nnzX[0],:]
		xSubs = xSubs[nnzX[0],:]
		X.append(sptensor.sptensor(xSubs, xVals, M.shape))
	## return the observation
	return X

L, A, U = generateSolution(4,5,[100,80,60,50],[20,15,10,5],2)
MFull = generateOriginalTensor(L,A,U,[[0,1,2], [0,1,3], [0,2,3]], 20)
X = generateRandomProblem(MFull)