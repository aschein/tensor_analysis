"""
Experiment to compute the predictive accuracy of Marble

Arguments
--------------
expt:	the unique id for this set of experiments
r:		rank of decomposition
alpha:	the weight of the bias tensor			
ms:		(optional) the size of each dimension
fl: 	(optional) the non-zero percentage along each dimension
g:		(optional) the minimum non-zero entry value
s:		(optional) random seed value
"""
import argparse
import numpy as np
import time
import shelve
import json
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import sys
sys.path.append("..")

import KLProjection
import CP_APR
import SP_NTF
import predictionTools
import tensorIO

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="input file")
parser.add_argument("eid", type=int, help="experiment id")
parser.add_argument("rank", type=int, help="rank to evaluate")
parser.add_argument("iter", type=int, help="the number of outer iterations")
parser.add_argument("alpha", type=float, help="alpha")
parser.add_argument("-t", "--testSize", type=float, help="test size", default=0.5)
parser.add_argument("-g", '--gamma', nargs='+', type=float, help="gamma")
parser.add_argument("-s", "--seed", type=int, help="random seed", default=0)
args = parser.parse_args()

inputFile = args.infile
exptID = args.eid
testSize = args.testSize
innerIter = 10
outerIter = args.iter
R = args.rank
gamma = args.gamma
alpha = args.alpha
seed = args.seed

X, axisDict, classDict = tensorIO.loadSingleTensor(inputFile)
Y = np.array(classDict.values(), dtype='int')
ttss = StratifiedShuffleSplit(Y, n_iter=1, test_size=testSize, random_state=seed)
predModel = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

output = { "expt": exptID, "iters": outerIter, "inner": innerIter, "R": R, 
	"gamma": gamma, "alpha": alpha, "seed": seed }

for train, test in ttss:
	trainShape = list(X.shape)
	trainShape[0] = len(train)
	## take the subset for training
	trainX = predictionTools.tensorSubset(X, train, trainShape)
	trainY = Y[train]

	## create the raw features
	rawFeatures = predictionTools.createRawFeatures(X)
	startTime = time.time()
	MFact, Minfo = SP_NTF.sp_ntf(trainX, R=R, alpha=alpha, gamma=gamma, maxiters = outerIter, maxinner=innerIter)
	marbleElapse = time.time() - startTime
	pftMat, pftBias = SP_NTF.projectTensor(X, MFact, 0, maxinner=innerIter)
	
	## store off the raw file
	MFact[0].writeRawFile("results/pred-raw-marble-{0}.dat".format(exptID))
	MFact[1].writeRawFile("results/pred-raw-bias-marble-{0}.dat".format(exptID))

	## compare to the traditional non-negative
	startTime = time.time()
	MCPR, cpstats, mstats = CP_APR.cp_apr(trainX, R, maxiters=outerIter, maxinner=innerIter)
	cpaprElapse = time.time() - startTime
	MCPR.writeRawFile("results/pred-raw-cpapr-{0}.dat".format(exptID))
	MCPR.normalize_sort(1)
	klp = KLProjection.KLProjection(MCPR.U, MCPR.R)
	cprFeat = klp.projectSlice(X, 0)

	## prediction part
	baseAUC, basePred = predictionTools.getAUC(predModel, rawFeatures, Y, train, test)
	marbleAUC, marblePred = predictionTools.getAUC(predModel, pftMat, Y, train, test)
	cprAUC, cprPred = predictionTools.getAUC(predModel, cprFeat, Y, train, test)

	output['time'] = [0, cpaprElapse, marbleElapse]
	output['auc'] = [baseAUC, cprAUC, marbleAUC]
	output['order'] = ['Baseline', 'CP-APR', 'Marble']

with open("results/pred-{0}.json".format(exptID), 'w') as outfile:
	json.dump(output, outfile)