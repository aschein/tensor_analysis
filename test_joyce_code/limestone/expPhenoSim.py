import numpy as np
import nimfa
from sklearn import preprocessing
from scipy import sparse
import argparse
import itertools

import decompTools
import sptensor
import sptenmat
import khatrirao
import CP_APR

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="input file")
parser.add_argument("expt", type=int, help="experiment id offset")
parser.add_argument("-r", "--rank", type=int, help="rank of factorization", default=40)
parser.add_argument("-s", "--seed", type=int, help="random seed", default=0)
parser.add_argument("-i", "--iterations", type=int, help="Number of outer interations", default=70)
args = parser.parse_args()

R = args.rank
seed = args.seed
iters = args.iterations
filename = args.inputFile
exptID = args.expt

innerIter = 10
patThresh = 1e-50
modeThr = 1e-2

X = sptensor.loadTensor(filename.format("data"))
yaxis = decompTools.loadAxisInfo(filename.format("info"))

## calculate diagnosis-medication combination
diagMed = [[a, b] for a, b in itertools.product(yaxis[1], yaxis[2])] 

def getDBEntry(featureName, m):
    output = np.zeros((1, 4))
    for r in range(R):
        # get the nonzero indices
        idx = np.flatnonzero(m[:, r])
        tmp = np.column_stack((np.array(diagMed)[idx], np.repeat(r, len(idx)), m[idx, r]))
        output = np.vstack((output, tmp))
    output = np.delete(output, (0), axis=0)
    output = np.column_stack((np.repeat(exptID, output.shape[0]), np.repeat(featureName, output.shape[0]), output))
    return output

np.random.seed(seed)
M, cpstats, mstats = CP_APR.cp_apr(X, R, maxiters=iters, maxinner=innerIter)
M.normalize_sort(1)
## Threshold the values
for n in range(1,2):
    zeroIdx = np.where(M.U[n] < modeThr)
    M.U[n][zeroIdx] = 0
## Get the diagnosis-medication matrix
ptfMatrix = khatrirao.khatrirao(M.U[1], M.U[2])
dbOutput = getDBEntry("CP-APR", ptfMatrix)

flatX = sptenmat.sptenmat(X, [0]).tocsrmat() # matricize along the first mode
nmfModel = nimfa.mf(flatX, method="nmf", max_iter=iters, rank=R)
nmfResult = nimfa.mf_run(nmfModel)
nmfBasis = nmfResult.coef().transpose()
nmfBasis = preprocessing.normalize(nmfBasis, norm="l1", axis=0)
nmfBasis = nmfBasis.toarray()
zeroIdx = np.where(nmfBasis < modeThr*modeThr)
nmfBasis[zeroIdx]= 0
dbOutput = np.vstack((dbOutput, getDBEntry("NMF", nmfBasis)))

## write the DBOutput
Youtfile = "results/sim-db-{0}.csv".format(exptID)
np.savetxt(Youtfile, dbOutput, fmt="%s", delimiter="|")

def greedy_fms(A, B):
    A = preprocessing.normalize(A, norm="l2", axis=0)
    B = preprocessing.normalize(B, norm="l2", axis=0)
    C = abs(np.dot(A.transpose(), B))
    AR = []
    BR = []
    score = []
    for r in range(R):
        maxIdx = np.unravel_index(C.argmax(), C.shape)
        AR.append(maxIdx[0])
        BR.append(maxIdx[1])
        score.append(C[maxIdx])
        C[maxIdx[0],:] = 0
        C[:, maxIdx[1]] = 0
    return np.column_stack((np.repeat(exptID, R), np.array(AR), np.array(BR), np.array(score)))

## write the score output
yScorefile = "results/sim-score-{0}.csv".format(exptID)
scoreOutput = greedy_fms(ptfMatrix, nmfBasis)
np.savetxt(yScorefile, scoreOutput, fmt="%s", delimiter="|")

Ysqlfile = "results/sim-sql-{0}.sql".format(exptID)
sqlOut = file(Ysqlfile, "w")
sqlOut.write("load data local infile '/home/joyce/workspace/Health/analysis/tensor/{0}' into table sim_factors fields terminated by '|'  ;\n".format(Youtfile))
sqlOut.write("load data local infile '/home/joyce/workspace/Health/analysis/tensor/{0}' into table sim_metrics fields terminated by '|'  ;\n".format(yScorefile))
sqlOut.write("insert into sim_models(expt_ID, rank, iterations, seed) values({0}, {1}, {2}, {3});\n".format(exptID, R, iters, seed))
sqlOut.close()