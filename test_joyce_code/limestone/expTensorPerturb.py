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
import argparse
import json
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
R = args.rank
seed = args.seed
outerIters = args.iterations
innerIters = 10
tol = 1e-2
zeroThr = 1e-50

noiseParam = 2
noisePercent = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

# input file and output file
inputFile = args.inputFile.format("data")
X = sptensor.loadTensor(inputFile)

def factorTensor(X):
    # set the seed for the same initialization
    np.random.seed(seed)
    Y, iterStats, mstats = CP_APR.cp_apr(X, R, tol=tol, maxiters=outerIters, maxinner=innerIters)
    Y.normalize_sort(1)
    Y = decompTools.zeroSmallFactors(Y, zeroThr=zeroThr)
    return Y

# do the first one
baseTF = factorTensor(X)

totNonzero = len(X.vals)
scoreResults = np.empty((1,7), dtype="S20")

outfile = open("results/perturb-type1-{0}.json".format(exptID), 'w') 
# now we want to do the others
for noise in noisePercent:
    # figure out what percentage should be disrupted
    noiseNum = int(totNonzero*noise)
    noiseVals = np.random.poisson(lam=noiseParam, size=noiseNum)
    noiseSubs = np.random.randint(low=0, high=totNonzero, size=noiseNum)
    Y = X.copy()
    Y.vals[noiseSubs, 0] = Y.vals[noiseSubs, 0] + noiseVals
    noiseTF = factorTensor(Y)
    fms = baseTF.greedy_fms(noiseTF)

    outfile.write(json.dumps({"expt": exptID, "type": "add", "noise": noise, 
        "seed": seed, "rank": R, "0": fms['0'], "1": fms['1'], "2": fms['2']}) + "\n")

outfile.close()
