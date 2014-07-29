"""
Experiment to compare Marble with CP-APR

Arguments
--------------
expt:	the unique id for this set of experiments
r:		rank of decomposition
alpha:	the weight of the bias tensor			
ms:		(optional) the size of each dimension
fl: 	(optional) the non-zero percentage along each dimension
g:		(optional) the minimum non-zero entry value
"""
import json
import argparse
import time
import numpy as np
import sys
sys.path.append("..")

import ktensor
import tensorTools
import SP_NTF
import CP_APR

import simultTools

parser = argparse.ArgumentParser()
parser.add_argument("expt", type=int, help="experiment number")
parser.add_argument("r", type=int, help="rank")
parser.add_argument("alpha", type=int, help="alpha")
parser.add_argument('-ms','--MS', nargs='+', type=int, help="size of matrix") 
parser.add_argument("-fl", '--fill', nargs='+', type=int, help="percentage of nonzeros")
parser.add_argument("-g", '--gamma', nargs='+', type=float, help="gamma")
args = parser.parse_args()

exptID = args.expt
R = args.r
alpha = args.alpha
MSize = args.MS
gamma = args.gamma
AFill = args.fill
INNER_ITER = 5
MAX_ITER = 500

print "Generating simulation data with known decomposition"
## generate the solution
TM, TMHat = simultTools.generateSolution(MSize, R, AFill, alpha)
TMFull = TM.toTensor() + TMHat.toTensor()
## generate an observation from the known solution
X = simultTools.generateRandomProblem(TMFull)

data = {'exptID': exptID, 'size': MSize, 'sparsity': AFill, "rank": R, "alpha": alpha, "gamma": gamma}

def calculateValues(TM, M):
	fms = TM.greedy_fms(M)
	fos = TM.greedy_fos(M)
	nnz = tensorTools.countTensorNNZ(M)
	return fms, fos, nnz

for sample in range(10):
	seed = sample*1000
	np.random.seed(seed)
	## solve the solution
	startTime = time.time()
	spntf = SP_NTF.SP_NTF(X, R=R, alpha=alpha, maxinner=INNER_ITER, maxiters=MAX_ITER)
	Yinfo = spntf.computeDecomp(gamma=gamma)
	## calculate all the request entries
	marbleElapse = time.time() - startTime
	marbleFMS, marbleFOS, marbleNNZ = calculateValues(TM, spntf.M[SP_NTF.REG_LOCATION])

	np.random.seed(seed)
	startTime = time.time()
	YCP, ycpstats, mstats = CP_APR.cp_apr(X, R=R, maxinner=INNER_ITER, maxiters=MAX_ITER)
	cpaprElapse = time.time() - startTime
	cpaprFMS, cpaprFOS, cpaprNNZ = calculateValues(TM, YCP)

	for n in range(YCP.ndims()):
		YCP.U[n] = tensorTools.hardThresholdMatrix(YCP.U[n], gamma[n])
	limestoneFMS, limestoneFOS, limestoneNNZ = calculateValues(TM, YCP)

	sampleResult = {
	"Order": ["Marble", "CPAPR", "Limestone"],
	"FMS":[marbleFMS, cpaprFMS, limestoneFMS],
	"FOS":[marbleFOS, cpaprFOS, limestoneFOS],
	"CompTime": [marbleElapse, cpaprElapse, cpaprElapse],
	"NNZ": [marbleNNZ, cpaprNNZ, limestoneNNZ]
	}
	data[str(sample)] = sampleResult

with open('results/simulation-{0}.json'.format(exptID), 'w') as outfile:
	json.dump(data, outfile)