import numpy as np
import random

import sys
sys.path.append("..")
import sptensor
import ktensor

def generateSolution(sz, R, AFill, LambdaHat):
	A = []
	for n in range(len(sz)):
		A.append(np.zeros((sz[n], R)))
		for r in range(R):
			# randomly select some entries to be nonzero
			nnz = random.sample(range(sz[n]), AFill[n]) #selects AFill[n] elements among the array range(sz(n))
			A[n][nnz, r] = np.random.random(size=AFill[n])
			# percentage of large size
			bigSamp = int (0.1*sz[n])
			if bigSamp > AFill[n]:
				bigSamp = 1
			big = random.sample(nnz, bigSamp)
			A[n][big, r] = 10 * A[n][big, r]
	lmbda = np.random.random_integers(low = 1, high = 1, size=R)
	M = ktensor.ktensor(lmbda, A)
	M.normalize_sort(1)
	## generate the noise bias
	U = []
	for n in range(len(sz)):
		U.append(np.zeros((sz[n], 1)))
		U[n][:, 0] = np.random.random(size=sz[n])
	Mhat = ktensor.ktensor(np.array([1]), U)
	Mhat.normalize(1)
	Mhat.lmbda[0] = LambdaHat
	return M, Mhat

def generateRandomProblem(MFull):
	## calculate the two together
	nnz = np.nonzero(MFull.data)
	mfVals = MFull.data.flatten()
	xVals = np.reshape([np.random.poisson(l) for l in mfVals], (len(mfVals), 1))
	Xsubs = np.zeros((len(mfVals), MFull.ndims()))
	Xsubs.dtype = 'int'
	for n in range(MFull.ndims()):
		Xsubs[:, n] = nnz[n]
	X = sptensor.sptensor(Xsubs, xVals, MFull.shape)
	## return the observation
	return X