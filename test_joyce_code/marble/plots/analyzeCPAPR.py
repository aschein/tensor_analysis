import numpy as np
import json
import sys
sys.path.append("../..")

import ktensor
import tensorTools

dataOut = []
thr = [[1e-4], [1e-3, 1e-2, 2e-2, 5e-2, 1e-1], [1e-3, 1e-2, 2e-2, 5e-2, 1e-1]]

####### SAVE OFF THRESHOLD VALUES ############
for run in range(400, 410):
	M = ktensor.loadTensor("../results/pred-raw-cpapr-{0}.dat".format(run))
	for n in range(M.ndims()):
		for k in thr[n]:
			factMatrix = np.array(M.U[n])
			truncIdx = np.where(factMatrix < k)
			factMatrix[truncIdx] = 0
			## convert these to 1 to sum
			nnzIdx = np.nonzero(factMatrix)
			factMatrix[nnzIdx] = 1
			nnzVals = np.sum(factMatrix, axis=0)
			dataOut.append({"expt": run, "mode": n, "thr": k, "count": nnzVals.tolist()})

with open("bih-hist.json", 'w') as outfile:
	json.dump(dataOut, outfile)

###### TIME TO ANALYZE SOME FACTORS ########
def loadJSON(fn):
	with open(fn, 'rb') as outfile:
		jsonDict = json.load(outfile)
		outfile.close()
	return jsonDict

cptLevel = loadJSON("../data/cpt-level2.json")
icdLevel = loadJSON("../data/icd-level2.json")
X, axisDict, classDict = tensorTools.loadSingleTensor("../data/cms-tensor-{0}.dat")

## lookup values
def lookupDict(idx, n, axisDict, levelDict):
	ivAxis = {v: k for k, v in axisDict[n].items()}
	modeCat = [levelDict[str(ivAxis[k])] for k in idx]
	return modeCat

## looking for chronic diseases
def getLargeElements(MF, n, r, axisDict, levelDict, thr):
	nnzIdx = np.where(MF.U[n][:, r] > thr)
	modeCat = lookupDict(nnzIdx[0], n, axisDict, levelDict)
	vals = MF.U[n][nnzIdx,r]
	return modeCat, vals.tolist()

def getDisease(idx, chronicOut, disease, M, thr):
	hfIdx = axisDict[1][idx]
	mDiagR = np.flatnonzero(M.U[1][hfIdx,:] > 0.3)
	for r in mDiagR:
		mc1, mc1vals  = getLargeElements(M, 1, r, axisDict, icdLevel, thr[0])
		## argsort decreasing
		sortIdx = np.argsort(mc1vals, axis=None)[::-1]
		mc1 = np.array(mc1)[sortIdx].tolist()
		mc2, mc2vals  = getLargeElements(M, 2, r, axisDict, cptLevel, thr[1])
		sortIdx = np.argsort(mc2vals, axis=None)[::-1]
		mc2 = np.array(mc2)[sortIdx].tolist()
		nppl = float(np.flatnonzero(M.U[0][:,r] > 1e-4).shape[0]) / M.U[0].shape[0]
		chronicOut.append({"Disease": disease, "R": r, "Patient": nppl, "Diagnosis": mc1, "Procedure": mc2})
	return chronicOut

run = 404
M = ktensor.loadTensor("../results/pred-raw-cpapr-{0}.dat".format(run))

chronicOut = []
chronicOut = getDisease(52, chronicOut, "Hypertension", M, [0.05, 0.05])
## heart failure = 55
chronicOut = getDisease(55, chronicOut, "HF", M, [0.05, 0.05])
## diabetes = 30
chronicOut = getDisease(30, chronicOut, "Diabetes", M, [0.05, 0.05])
## arthritis = 88
chronicOut = getDisease(88, chronicOut, "Arthritis", M, [0.05, 0.05])

with open("bih-pheno.json", 'w') as outfile:
	json.dump(chronicOut, outfile)