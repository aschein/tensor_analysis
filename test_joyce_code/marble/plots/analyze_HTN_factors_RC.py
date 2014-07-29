import numpy as np
import json
import sys
from collections import OrderedDict
sys.path.append("../..")

import ktensor
import tensorTools

def loadJSON(fn):
	with open(fn, 'rb') as outfile:
		jsonDict = json.load(outfile)
		outfile.close()
	return jsonDict


MBias = ktensor.loadTensor("../results/pred-raw-bias-marble-{0}.dat".format(run))
M = ktensor.loadTensor("../results/pred-raw-marble-{0}.dat".format(run))
MCP = ktensor.loadTensor("../results/pred-raw-cpapr-{0}.dat".format(run))

X, axisDict, classDict = tensorTools.loadSingleTensor("../data/cms-tensor-{0}.dat")

cptLevel = loadJSON("../data/cpt-level2.json")
icdLevel = loadJSON("../data/icd-level2.json")

## lookup values
def lookupDict(idx, n, axisDict, levelDict):
	ivAxis = {v: k for k, v in axisDict[n].items()}
	modeCat = [levelDict[str(ivAxis[k])] for k in idx]
	return modeCat

## get the top k from MBias
def getTopK(MF, n, axisDict, levelDict, k = 10):
	sortIdx = np.argsort(MF.U[n], axis=None)[::-1][:k]
	modeCat = lookupDict(sortIdx, n, axisDict, levelDict)
	vals = MF.U[n][sortIdx,0]
	return OrderedDict(zip(modeCat, vals))

biasOut = []
biasOut.append(getTopK(MBias, 1, axisDict, icdLevel))
biasOut.append(getTopK(MBias, 2, axisDict, cptLevel))
with open("biasOut.json", 'w') as outfile:
	json.dump(biasOut, outfile)

M.normalize_sort(1)
## we want the congruences
M.normalize(2)
MCP.normalize(2)
# compute the product for each pair of matrices for each mode
rawC = [abs(np.dot(M.U[n].transpose(),MCP.U[n])) for n in range(M.ndims())]
C = np.ones((M.R, M.R))
for n in range(M.ndims()):
    C = C * rawC[n]

mcpIdx = np.argmax(C[:,0])

## renormalize M and MCCP
M.normalize(1)
MCP.normalize(1)

def getNonzero(MF, n, r, axisDict, levelDict):
	nnzIdx = np.flatnonzero(MF.U[n][:, r])
	modeCat = lookupDict(nnzIdx, n, axisDict, levelDict)
	vals = MF.U[n][nnzIdx,r]
	return modeCat, vals.tolist()

compOut = []
mc1, mc1vals = getNonzero(M, 1, 0, axisDict, icdLevel)
mc2, mc2vals = getNonzero(M, 2, 0, axisDict, cptLevel)

## truncate the entries a bit
truncIdx = np.where(MCP.U[1][:,mcpIdx] < 1e-10)
MCP.U[1][truncIdx,mcpIdx] = 0
truncIdx = np.where(MCP.U[2][:,mcpIdx] < 1e-10)
MCP.U[2][truncIdx,mcpIdx] = 0

cp1, cp1vals = getNonzero(MCP, 1, mcpIdx, axisDict, icdLevel)
cp2, cp2vals = getNonzero(MCP, 2, mcpIdx, axisDict, cptLevel)

compOut = []
compOut.append({"Model": "Marble", "Mode": 1, "cat": mc1, "values": mc1vals})
compOut.append({"Model": "Marble", "Mode": 2, "cat": mc2, "values": mc2vals})
compOut.append({"Model": "CPAPR", "Mode": 1, "cat": cp1, "values": cp1vals})
compOut.append({"Model": "CPAPR", "Mode": 2, "cat": cp2, "values": cp2vals})

with open('comp-cms-results.json', 'w') as outfile:
	json.dump(compOut, outfile)

## looking for chronic diseases
def getLargeElements(MF, n, r, axisDict, levelDict):
	nnzIdx = np.where(MF.U[n][:, r] > 0.05)
	modeCat = lookupDict(nnzIdx[0], n, axisDict, levelDict)
	vals = MF.U[n][nnzIdx,r]
	return modeCat, vals.tolist()

def getDisease(idx, chronicOut, disease):
	hfIdx = axisDict[1][idx]
	mDiagR = np.flatnonzero(M.U[1][hfIdx,:] > 0.3)
	for r in mDiagR:
		mc1, mc1vals  = getLargeElements(M, 1, r, axisDict, icdLevel)
		## argsort decreasing
		sortIdx = np.argsort(mc1vals, axis=None)[::-1]
		mc1 = np.array(mc1)[sortIdx].tolist()
		mc2, mc2vals  = getLargeElements(M, 2, r, axisDict, cptLevel)
		sortIdx = np.argsort(mc2vals, axis=None)[::-1]
		mc2 = np.array(mc2)[sortIdx].tolist()
		chronicOut.append({"Disease": disease, "R": r, "Diagnosis": mc1, "Procedure": mc2})
	return chronicOut

chronicOut = []
## hypertension
chronicOut = getDisease(52, chronicOut, "HF")
## heart failure = 55
chronicOut = getDisease(55, chronicOut, "HF")
## diabetes = 30
chronicOut = getDisease(30, chronicOut, "Diabetes")
## arthritis = 88
chronicOut = getDisease(88, chronicOut, "Arthritis")
with open('cms-chronic-results.json', 'w') as outfile:
	json.dump(chronicOut, outfile)

nmfBasisFile = file("../results/nmf-404.dat", "rb")
nmfBasis = np.load(nmfBasisFile)
nmfBasisFile.close()

def getFlatFeature(idx, axisDict):
    diagIdx = idx / len(axisDict[2])
    medIdx = idx % len(axisDict[2])
    return axisDict[1][diagIdx], axisDict[2][medIdx]