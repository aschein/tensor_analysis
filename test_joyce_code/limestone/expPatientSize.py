"""
Experiment to evaluate the effect of the size on computation time
"""
import time
import numpy as np
from sklearn.decomposition import RandomizedPCA
import nimfa
import argparse

import CP_APR
import sptensor
import sptenmat

# Load the original data
filename = 'data/hf-tensor-level1-data.dat'
X = sptensor.loadTensor(filename)
R = 40
iters=70
samples=10

pcaModel = RandomizedPCA(n_components=R)
stats = np.zeros((1, 6))

parser = argparse.ArgumentParser()
parser.add_argument("pat", type=int, help="number of patients")
args = parser.parse_args()
pn = args.pat

patList = np.arange(pn)
ix = np.in1d(X.subs[:,0].ravel(), patList)
idx = np.where(ix)[0]
xprime = sptensor.sptensor(X.subs[idx, :], X.vals[idx], [pn, X.shape[1], X.shape[2]])
flatX = sptenmat.sptenmat(xprime, [0]).tocsrmat() # matricize along the first mode
stats = np.zeros((1,6))

## NMF Timing
for k in range(samples):
    startTime = time.time()
    nmfModel = nimfa.mf(flatX, method="nmf", max_iter=iters, rank=R)
    nmfResult = nimfa.mf_run(nmfModel)
    elapsed = time.time() - startTime
    stats = np.vstack((stats, np.array([R, iters, pn, k, "NMF", elapsed])))
    
## PCA Timing
for k in range(samples):
    startTime = time.time()
    pcaModel.fit(flatX)
    elapsed = time.time() - startTime
    stats = np.vstack((stats, np.array([R, iters, pn, k, "PCA", elapsed])))

## Tensor factorization timing
for k in range(samples):
    startTime = time.time()
    CP_APR.cp_apr(xprime, R, maxiters=iters)
    elapsed = time.time() - startTime
    stats = np.vstack((stats, np.array([R, iters, pn, k, "CP_APR", elapsed])))
    
stats = np.delete(stats, (0), axis=0)

outFile = "results/patient-cpu-{0}.csv".format(pn)
np.savetxt(outFile, stats,  fmt="%s", delimiter="|")
print "load data local infile '/home/joyce/workspace/Health/analysis/tensor/{0}' into table comp_metrics fields terminated by '|'  ;\n".format(outFile)
