#run a test case
import os
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




os.chdir("/Users/localadmin/tensor_factorization/test_joyce_code/marble")

## code from compareIterations.py
exptID = 3
R = 3
alpha = 1
MSize = [2,2,2]
gamma = None
AFill = [2,2,2]
startSeed = 1

print "Generating simulation data with known decomposition"
TM, TMHat = simultTools.generateSolution(MSize, R, AFill, alpha)
TMFull = TM.toTensor() + TMHat.toTensor()
np.random.seed(startSeed)



#generate random problem
sample = 0
X = simultTools.generateRandomProblem(TMFull)
data = {'exptID': exptID, 'size': MSize, 'sparsity': AFill, 'sample': sample,
		"rank": R, "alpha": alpha, "gamma": gamma, "seed": startSeed}
## solve the solution
innerIt = 1
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