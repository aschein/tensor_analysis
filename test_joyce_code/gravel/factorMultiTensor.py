"""
Simultaneous tensor factorization experiment

Arguments
--------------
inputFile -    tensor input file format with {0} for the 2 separate files
expt -         the experiment number for these sets of parameters
label -        the patient cohort (all case, control or mixed)
description -  the patient set for the database table
exptDescription - the experiment description for the database table
iterations -   optional maximum number of outer iterations defaulting to 100
seed -         optional argument to set the seed for repeatable factorizations defaults to 0
rank -         optional the decomposition rank for the factorization defaulting to 100
"""
import numpy as np
import argparse
from pymongo import MongoClient
import copy

import sys
sys.path.append("..")
import tensorTools
import sim_APR

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="input file to parse")
parser.add_argument("expt", type=int, help="experiment number")
parser.add_argument("description", help="description of patient set")
parser.add_argument("-r", "--rank", type=int, help="rank of factorization", default=100)
parser.add_argument("-s", "--seed", type=int, help="random seed", default=0)
parser.add_argument("-i", "--iterations", type=int, help="Number of outer interations", default=100)
args = parser.parse_args()

## experimental setup
inputFile = args.inputFile
exptID = args.expt
patientSet = args.description
R = args.rank
seed = args.seed
outerIter = args.iterations
innerIter = 10
tol = 1e-2
zeroThr = 1e-4

## connection to mongo-db
client = MongoClient()
db = client.gravel
exptDB = db.factor

## verify the experimentID is okay
if exptDB.find({"id": exptID}).count():
	print "Experiment ID already exists, select another"
	return

print "Starting Simultaneous Tensor Factorization with ID:{0}".format(exptID)
X, sharedModes, axisDict, patClass = tensorTools.loadMultiTensor(inputFile)
## special case for single
if len(X) == 1:
	sharedModes = []

np.random.seed(seed)
sapr = sim_APR.SAPR(X, R, sharedModes, tol=tol, maxiters=outerIter, maxinner = innerIter)
M, cpstats, mstats = sapr.factorize()
## now we can construct the experimental results
mongoRow = {
	"id": exptID, 
	"seed": seed, 
	"description": description, 
	"rank": R, 
	"maxInner": innerIter, 
	"maxIter": mstats['Iters'],
	"logLikelihood": mstats['LL'],
	"kktViolations": mstats['KKT']
	}
## hard theshhold
allButOne = sapr.modes[1:]
firstMode = [sapr.modes[0]]
M = tensorTools.hardThresholdFactors(M, allButOne, 1e-2)
M = tensorTools.hardThresholdFactors(M, firstMode, 1e-6)
mongoRow["factors"] = tensorTools.getMultiMongoFormat(M, sapr.modes, axisDict)

multiTensorTools.saveRawFile(M, "results/multi-raw-{0}.dat".format(exptID))

## zero small factors out
M = multiTensorTools.hardThreshold(M, zeroThr)

Youtfile = "results/multi-db-{0}-{1}.csv".format(exptID, iter)
Ysqlfile = "results/multi-sql-{0}.sql".format(exptID)
Yout = multiTensorTools.getDBFormat(sapr.M, sapr.modes, axisDict)
Yout = np.column_stack((np.repeat(exptID, Yout.shape[0]), Yout))
np.savetxt(Youtfile, Yout, fmt="%s", delimiter="|")

sqlOut = file(Ysqlfile, "w")
sqlOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_factors;\n".format(Youtfile))
sqlOut.write("insert into joyceho.tensor_models values({0}, {1}, \'{2}\',\'{3}\', {4}, {5}, {6}, {7}, {8});\n".format(exptID, labelID, patientSet, exptDesc, iter, innerIter, mstats['LS'], mstats['LL'], mstats['KKT']))
sqlOut.close()

print "Completed Simultaneous Factorization with ID:{0}".format(exptID)
print "Import sql file " + Ysqlfile
