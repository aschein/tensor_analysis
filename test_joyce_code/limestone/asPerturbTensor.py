"""
Regular tensor factorization experiment

Command line parameters
------------------------
Required:
inputFile of the style: <filestart>-{0}.dat from which the data and the axis information is derived
expt: experimental ID
label: the label ID for patients
description: describe the patient set for the sql file
-r : rank of the tensor factorizaiton
-s : the random seed for repeatability
-i : the number of outer iterations
"""
import numpy as np
import json
import argparse

import sys
sys.path.append("..")

import decompTools
import sptensor
import CP_APR

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--inputFile", help="input file to parse", default="data/hf-tensor-case-{0}.dat")
parser.add_argument("expt", type=int, help="experiment number")
parser.add_argument("-r", "--rank", type=int, help="rank of factorization", default=50)
parser.add_argument("-s", "--seed", type=int, help="random seed", default=0)
parser.add_argument("-i", "--iterations", type=int, help="Number of outer interations", default=70)
args = parser.parse_args()

## experimental setup
exptID = args.expt
inFile = args.inputFile
R = args.rank
seed = args.seed
outerIters = args.iterations
innerIters = 10
tol = 1e-2
zeroThr = 1e-10

noiseParam = 2
noisePercent = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

# input file and output file
inputFile = inFile.format("data")
X = sptensor.loadTensor(inputFile)

def factorTensor(X):
    # set the seed for the same initialization
    np.random.seed(seed)
    Y, iterStats, mstats = CP_APR.cp_apr(X, R, tol=tol, maxiters=outerIters, maxinner=innerIters)
    Y.normalize_sort(1)
    Y = decompTools.zeroSmallFactors(Y, zeroThr=zeroThr)
    return Y

print "Starting Tensor Factorization with ID:{0}".format(exptID)
# compute the base comparison
baseTF = factorTensor(X)
totNonzero = len(X.vals)

outfile = open("results/perturb-{0}.json".format(exptID), 'w') 
# now we want to do the others
for noise in noisePercent:
    # figure out what percentage should be disrupted
    noiseNum = int(totNonzero*noise)
    noiseVals = np.random.poisson(lam=noiseParam, size=noiseNum)
    noiseSubs = np.random.randint(low=0, high=totNonzero, size=noiseNum)
    ## first choose a number between 0 and 1 to denote add or subtract
    noiseOp = np.random.randint(low=0, high=2, size=noiseNum)
    addIdx = np.where(noiseOp == 0)[0]
    Y = X.copy()
    Y.vals[noiseSubs[addIdx], 0] = Y.vals[noiseSubs[addIdx], 0] + noiseVals[addIdx]
    ## do the subtraction
    subtractIdx = np.where(noiseOp == 1)[0]
    Y.vals[noiseSubs[subtractIdx], 0] = Y.vals[noiseSubs[subtractIdx], 0] - noiseVals[subtractIdx]
    ## anything that was zero-ed out we want to fix
    nozIdx = np.where(Y.vals <= 0)[0]
    Y.vals[nozIdx] = 0
    ## then we will add more by sampling empty space
    nozVals = np.random.poisson(lam=1, size=len(nozIdx)).reshape(len(nozIdx), 1)
    nozVals[np.where(nozVals == 0)] = 1
    nozSub0 = np.random.randint(low=0, high=Y.shape[0], size=len(nozVals))
    nozSub1 = np.random.randint(low=0, high=Y.shape[1], size=len(nozVals))
    nozSub2 = np.random.randint(low=0, high=Y.shape[2], size=len(nozVals))
    nozSubs = np.column_stack((nozSub0, nozSub1, nozSub2))
    Y.subs = np.vstack((Y.subs, nozSubs))
    Y.vals = np.vstack((Y.vals, nozVals))
    Y = sptensor.sptensor(Y.subs, Y.vals, Y.shape)
    noiseTF = factorTensor(Y)
    fms = baseTF.greedy_fms(noiseTF)

    outfile.write(json.dumps({"expt": exptID, "type": "add+subtract", "noise": noise, 
        "seed": seed, "rank": R, "0": fms['0'], "1": fms['1'], "2": fms['2']}) + "\n")

outfile.close()