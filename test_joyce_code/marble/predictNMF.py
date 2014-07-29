"""
Experiment to compute the predictive model
"""
import argparse
import numpy as np
import shelve
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import preprocessing
import nimfa

import sys
sys.path.append("..")

import KLProjection
import predictionModel
import sptenmat
import tensorTools
import json

X, axisDict, classDict = tensorTools.loadSingleTensor("data/cms-tensor-{0}.dat")
Y = np.array(classDict.values(), dtype='int')

flatX =  sptenmat.sptenmat(X, [0]).tocsrmat() # matricize along the first mode
testSize = 0.5

seed = 400
R = 50
ttss = StratifiedShuffleSplit(Y, n_iter=1, test_size=testSize, random_state=seed)

for train, test in ttss:
	nmfModel = nimfa.mf(flatX[train,:], method="nmf", max_iter=200, rank=R)
	nmfResult = nimfa.mf_run(nmfModel)
	nmfBasis = nmfResult.coef().transpose()
	nmfBasis = preprocessing.normalize(nmfBasis, norm="l1", axis=0)
	nmfBasisA = nmfBasis.toarray()
	outFile = file("results/nmf-404.dat", "wb")
	np.save(outFile, nmfBasisA)
	outFile.close()