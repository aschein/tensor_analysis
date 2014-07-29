import numpy as np
from pymongo import MongoClient
from collections import OrderedDict
import sys
sys.path.append("../..")

import json
import ktensor
import tensorTools

mongoC = MongoClient()
db = mongoC.limestone
ppDB = db.predPower

dataOut = []
exptId = []

for k in ppDB.find({"R": 50}):
	tmp = {"expt": k["expt"], "comp": k["Comp"], "model": k["Model"]}
	dataOut.append(tmp)
	exptId.append(k["expt"])

with open('predPower.json', 'w') as outfile:
	json.dump(dataOut, outfile)

exptId = set(exptId)

thr = 0.05*0.05

def binarizeAndSum(featMat):
	nnzIdx = np.nonzero(featMat)
	featMat[nnzIdx] = 1
	return np.sum(featMat, axis=0)

dataOut = []
for expt in exptId:
	limestone = np.load("../results/pred-metric-limestone-{0}.dat.npy".format(expt))
	zeroIdx = np.nonzero(limestone < thr)
	limestone[zeroIdx] = 0
	nnzPheno = binarizeAndSum(limestone)
	tmp = {"expt": k["expt"], "model": "Limestone", "nnz": nnzPheno.tolist()}
	dataOut.append(tmp)
	nmf = np.load("../results/pred-metric-nmf-{0}.dat.npy".format(expt))
	nnzPheno = binarizeAndSum(nmf)
	tmp = {"expt": k["expt"], "model": "NMF", "nnz": nnzPheno.tolist()}
	dataOut.append(tmp)
	pca = np.load("../results/pred-metric-pca-{0}.dat.npy".format(expt))
	nnzPheno = binarizeAndSum(np.transpose(pca))
	tmp = {"expt": k["expt"], "model": "PCA", "nnz": nnzPheno.tolist()}
	dataOut.append(tmp)

with open('predNNZ.json', 'w') as outfile:
	json.dump(dataOut, outfile)