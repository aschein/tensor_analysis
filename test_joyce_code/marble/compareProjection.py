"""
Experiment to evaluate the effect of gradual projection, no projection, and full projection

Arguments
--------------
expt:	the unique id for this set of experiments
r:		rank of decomposition
alpha:	the weight of the bias tensor			
ms:		(optional) size of each dimension
fl: 	(optional) number of non-zeros along each dimension
g:		(optional) minimum non-zero entry value in solution
"""
import numpy as np
import random
import json
import argparse
import time
import sys
sys.path.append("..")

import SP_NTF
import tensorTools
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
EXPT_GAMMA = args.gamma
AFill = args.fill
INNER_ITER = 5


###  Generate the solution for comparison purposes ###
print "Generating simulation data with known decomposition!"

TM, TMHat = simultTools.generateSolution(MSize, R, AFill, alpha)
TMFull = TM.toTensor() + TMHat.toTensor()

np.random.seed(0)
outfile = open('results/projection-{0}.json'.format(exptID), 'w')

def calculateResult(gamma, seed, gradual=True):
	np.random.seed(seed)
	startTime = time.time()
	spntf = SP_NTF.SP_NTF(X, R=R, alpha=alpha, maxinner=INNER_ITER) # instance of SP_NTF class
	Yinfo = spntf.computeDecomp(gamma=gamma, gradual=gradual)
	totalTime = time.time() - startTime
	return {"gamma": gamma,
		"nnz": tensorTools.countTensorNNZ(spntf.M[0]),
		"compTime": totalTime,
		"iterInfo": Yinfo,
		"fms": TM.greedy_fms(spntf.M[0])}

for sample in range(1):
	## generate a random problem
	X = simultTools.generateRandomProblem(TMFull)
	data = {'exptID': exptID, 'size': MSize, 'sparsity': AFill, "rank": R, "alpha": alpha, "gamma": EXPT_GAMMA, 'sample': sample}
	seed = sample*1000
	data["none"] = calculateResult([0,0,0], seed)	## no projection
	data["full"] = calculateResult(EXPT_GAMMA, seed, gradual=False)	## hard projection
	data["gamma"] = calculateResult(EXPT_GAMMA, seed)	## gradual projection
	## soften gamma by factor of 1e-1
	tmpG = [g*1e-1 for g in EXPT_GAMMA]
	data["gamma2"] = calculateResult(tmpG, seed)
	## soften gamma by factor of 1e-2
	tmpG = [g*1e-2 for g in EXPT_GAMMA]
	data["gamma1"] = calculateResult(tmpG, seed)
	outfile.write(json.dumps(data) + "\n")

outfile.close()