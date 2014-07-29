"""
Experiment to compute the baseline predictive model using flat features
"""
import json
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("..")

import sptenmat
import tensorIO
import predictionTools

X, axisDict, classDict = tensorIO.loadSingleTensor("data/cms-tensor-{0}.dat")
Y = np.array(classDict.values(), dtype='int')
predModel = LogisticRegression(C=990000, penalty='l1', tol=1e-6)
flatX =  sptenmat.sptenmat(X, [0]).tocsrmat() # matricize along the first mode
testSize = 0.5

outfile = open("results/baseline-results.json", 'w')
for seed in range(0, 1000, 100):
	ttss = StratifiedShuffleSplit(Y, n_iter=1, test_size=testSize, random_state=seed)
	for train, test in ttss:
		trainY = Y[train]
		baseAUC, basePred = predictionTools.getAUC(predModel, flatX, Y, train, test)
		output = {"type": "baseline", "seed": seed, "auc": baseAUC }
		outfile.write(json.dumps(output) + '\n')

outfile.close()