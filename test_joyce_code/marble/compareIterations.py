"""
Experiment to evaluate the effect of the number of iterations based on simulated data

Arguments
--------------
expt:	the unique id for this set of experiments
r:		rank of decomposition
alpha:	the weight of the bias tensor			
ms:		(optional) size of each dimension
fl: 	(optional) number of non-zeros along each dimension
g:		(optional) minimum non-zero entry value in solution
s:		(optional) random seed value
"""
import numpy as np
import random
import json
import argparse
import time
import sys
sys.path.append("..")

import tensor
import SP_NTF
import simultTools

######## Parse the arguments passed in via the command line ##########
parser = argparse.ArgumentParser()
parser.add_argument("expt", type=int, help="experiment number")
parser.add_argument("r", type=int, help="rank")
parser.add_argument("alpha", type=int, help="alpha")
parser.add_argument('-ms','--MS', nargs='+', type=int, help="size of matrix") 
parser.add_argument("-fl", '--fill', nargs='+', type=int, help="percentage of nonzeros")
parser.add_argument("-g", '--gamma', nargs='+', type=float, help="gamma")
parser.add_argument("-s", '--seed', type=int, help="randomseed", default=0)
args = parser.parse_args()

exptID = args.expt
R = args.r
alpha = args.alpha
MSize = args.MS
gamma = args.gamma
AFill = args.fill
startSeed = args.seed

###  Generate the solution for comparison purposes ###
print "Generating simulation data with known decomposition"
TM, TMHat = simultTools.generateSolution(MSize, R, AFill, alpha)
TMFull = TM.toTensor() + TMHat.toTensor()
np.random.seed(startSeed)

outfile = open('results/iteration-{0}.json'.format(exptID), 'w')

for sample in range(10):
	## generate a random problem
	X = simultTools.generateRandomProblem(TMFull) #sparse tensor
	data = {'exptID': exptID, 'size': MSize, 'sparsity': AFill, 'sample': sample,
		"rank": R, "alpha": alpha, "gamma": gamma, "seed": startSeed}
	seed = sample*1000
	for innerIt in [1, 2, 5, 10]:
		## set seed for consistency
		np.random.seed(seed)
		## solve the solution
		startTime = time.time()
		spntf = SP_NTF.SP_NTF(X, R=R, alpha=alpha, maxinner = innerIt)
		Yinfo = spntf.computeDecomp(gamma=gamma)
		totalTime = time.time() - startTime
		sampleResult = {
			"compTime": totalTime,
			"iterInfo": Yinfo,
			"fms": TM.greedy_fms(spntf.M[SP_NTF.REG_LOCATION])
		}
		data[str(innerIt)] = sampleResult
	outfile.write(json.dumps(data) + '\n')

outfile.close()