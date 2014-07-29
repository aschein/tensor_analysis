"""
Experiment to compare the factorization using different ranks

Command line parameters
------------------------
Required:
inputFile of the style: <filestart>-{0}.dat from which the data and the axis information is derived
description: describe the patient set for the sql file
label: the label ID for patients
-e : optional parameter for experimental ID
-s : the random seed for repeatability
-i : the number of outer iterations
"""
import decompTools
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="input file to parse")
parser.add_argument("description", help="description of experiment")
parser.add_argument("label", type=int, help="label ID for patient set")
parser.add_argument("-e", "--expt", type=int, help="experiment number", default=-1)
parser.add_argument("-s", "--seed", type=int, help="random seed", default=0)
parser.add_argument("-i", "--iterations", type=int, help="Number of outer interations", default=100)
parser.add_argument("-rs", "--rankstart", type=int, help="Start of rank incrementation", default=25)
parser.add_argument("-re", "--rankend", type=int, help="End of rank increments", default=200)
args = parser.parse_args()

# Experimental sections
iter = args.iterations
innerIter = 10
tol = 1e-2
zeroThr = 1e-5
seed = args.seed
R = np.concatenate(([5, 10], np.arange(args.rankstart, args.rankend+25, 25)))
exptID = args.expt
labelID = args.label
exptDesc = args.description

inputFile = args.inputFile.format("data")
yaxis = decompTools.loadAxisInfo(args.inputFile.format("info"))

sqlOutfile = "results/rank-sql-{0}.sql".format(exptID)
rawFilePattern = "results/rank-raw-{0}-{1}.dat"
dbOutPattern = "results/rank-db-{0}-{1}.dat"
dbTimePattern = "results/rank-time-db-{0}-{1}.dat"

print "Starting Tensor Rank Experiment with ID:{0}".format(exptID)

sqlOut = file(sqlOutfile, "w")
for r in R:
    np.random.seed(seed)
    Y, ystats, mstats = decompTools.decomposeCountTensor(inputFile, R=r, outerIters=iter, innerIters=innerIter, convergeTol=tol, zeroTol=zeroThr)
    Y.writeRawFile(rawFilePattern.format(exptID, r)) # output the raw file
    # output the saved db file
    dbYFile = dbOutPattern.format(exptID, r)
    Yout = decompTools.getDBOutput(Y, yaxis)
    Yout = np.column_stack((np.repeat(exptID, Yout.shape[0]), Yout))
    np.savetxt(dbYFile, Yout, fmt="%s", delimiter="|")
    sqlOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_rank_factors;\n".format(dbYFile))
    # output the rank_times
    timeStats = np.delete(ystats, range(2,6), axis=1)
    timeStats = np.column_stack((np.repeat(exptID, timeStats.shape[0]), np.repeat(r, timeStats.shape[0]), timeStats))
    timeFile = dbTimePattern.format(exptID, r)
    np.savetxt(timeFile, timeStats, delimiter="|")
    sqlOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_rank_times;\n".format(timeFile))
    # output the model results
    sqlOut.write("insert into joyceho.tensor_rank_models values({0},{1},\'{2}\',{3},{4},{5},{6},{7},{8});\n".format(exptID, labelID, exptDesc, iter, innerIter, r, mstats['LS'], mstats['LL'], mstats['KKT']))

sqlOut.close()
print "Complete Tensor Rank Experiment with ID:{0}".format(exptID)
print "Import sql file " + sqlOutfile
