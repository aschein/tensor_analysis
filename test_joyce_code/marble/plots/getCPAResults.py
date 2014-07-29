import numpy as np
import json
import sys
sys.path.append("../..")

import ktensor
import tensorTools

dataOut = []
bins = [None, None, None]
for run in range(400, 409):
	M = ktensor.loadTensor("../results/pred-raw-cpapr-{0}.dat".format(run))
	for n in range(M.ndims()):
		factVals = M.U[n].flatten()
		nnzIdx = np.nonzero(factVals)
		factVals = factVals[nnzIdx]
		if bins[n] == None:
			factHist = np.histogram(factVals, bins = 10)
			bins[n] = factHist[1]
		else:
			factHist = np.histogram(factVals, bins = bins[n])
		dataOut.append({"expt": run, "mode": n, "count": factHist[0].tolist(), "bins": factHist[1].tolist()})

with open("cpapr-hist.json", 'w') as outfile:
	json.dump(dataOut, outfile)