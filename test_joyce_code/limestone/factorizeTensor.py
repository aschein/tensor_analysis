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
from pymongo import MongoClient
import sys
sys.path.append("..")

import tensorTools
import CP_APR

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="input file to parse")
parser.add_argument("expt", type=int, help="experiment number")
parser.add_argument("exptDescription", help="description of experiment")
parser.add_argument("-r", "--rank", type=int, help="rank of factorization", default=40)
parser.add_argument("-s", "--seed", type=int, help="random seed", default=0)
parser.add_argument("-i", "--iterations", type=int, help="Number of outer interations", default=100)
args = parser.parse_args()

## experimental setup
exptID = args.expt
exptDesc = args.exptDescription
R = args.rank
seed = args.seed
outerIters = args.iterations
innerIters = 10
tol = 1e-2

## load tensor information
X, axisDict, classDict = tensorTools.loadSingleTensor(args.inputFile)

## connection to mongo-db
client = MongoClient()
db = client.gravel
exptDB = db.factor

## verify the experimentID is okay
if exptDB.find({"id": exptID}).count():
	print "Experiment ID already exists, select another"
	return

print "Starting Tensor Factorization with ID:{0}".format(exptID)
np.random.seed(seed)

## factorize using CP_APR (this is the original)
Y, iterStats, modelStats = CP_APR.cp_apr(X, R, tol=tol, maxiters=outerIters, maxinner=innerIters)

##


Y.writeRawFile("results/apr-raw-{0}.dat".format(exptID))
Youtfile = "results/apr-db-{0}-{1}.csv".format(exptID, iter)
Ysqlfile = "results/apr-sql-{0}.sql".format(exptID)
# save the decomposition into the format
Yout = decompTools.getDBOutput(Y, yaxis)
Yout = np.column_stack((np.repeat(exptID, Yout.shape[0]), Yout))
np.savetxt(Youtfile, Yout, fmt="%s", delimiter="|")

sqlOut = file(Ysqlfile, "w")
sqlOut.write("load data local infile '/home/joyce/workspace//Health/analysis/tensor/{0}' into table tensor_factors fields terminated by '|'  ;\n".format(Youtfile))
sqlOut.write("insert into tensor_models(expt_ID, label_ID, description, rank, iterations, inner_iterations, seed, least_squares, log_likelihood, kkt_violation) values({0}, {1}, \'{2}\', {3}, {4}, {5}, {6}, {7}, {8}, {9});\n".format(exptID, labelID, exptDesc, R, iter, innerIter, seed, mstats['LS'], mstats['LL'], mstats['KKT']))
sqlOut.close()

print "Completed Tensor Factorization with ID:{0}".format(exptID)
print "Import sql file " + Ysqlfile
