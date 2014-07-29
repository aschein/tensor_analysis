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
import decompTools
import numpy as np
import argparse
import sptensor
import CP_ALS
import CP_APR

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="input file to parse")
parser.add_argument("expt", type=int, help="experiment number")
parser.add_argument("label", type=int, help="label ID for patient set")
parser.add_argument("description", help="description of patient set")
parser.add_argument("exptDescription", help="description of experiment")
parser.add_argument("-r", "--rank", type=int, help="rank of factorization", default=100)
parser.add_argument("-s", "--seed", type=int, help="random seed", default=0)
parser.add_argument("-i", "--iterations", type=int, help="Number of outer interations", default=100)
args = parser.parse_args()

## experimental setup
exptID = args.expt
labelID = args.label
patientSet = args.description
exptDesc = args.exptDescription
R = args.rank
seed = args.seed
iter = args.iterations
innerIter = 10
tol = 1e-2
zeroThr = 1e-5

# input file and output file
inputFile = args.inputFile.format("data")
yaxis = decompTools.loadAxisInfo(args.inputFile.format("info"))

print "Starting Tensor Factorization with ID:{0}".format(exptID)
X = sptensor.loadTensor(inputFile)
np.random.seed(seed)
Y, ls = CP_ALS.cp_als(X, R, tol=tol, maxiters=iter)
ll = CP_APR.loglikelihood(X, Y)

# normalize the factors using the 1 norm and then sort in descending order
Y.normalize_sort(1)
Y = decompTools.zeroSmallFactors(Y, zeroThr=zeroThr)

Y.writeRawFile("results/als-raw-{0}.dat".format(exptID))
Youtfile = "results/als-db-{0}-{1}.csv".format(exptID, iter)
Ysqlfile = "results/als-sql-{0}.sql".format(exptID)
# save the decomposition into the format
Yout = decompTools.getDBOutput(Y, yaxis)
Yout = np.column_stack((np.repeat(exptID, Yout.shape[0]), Yout))
np.savetxt(Youtfile, Yout, fmt="%s", delimiter="|")

sqlOut = file(Ysqlfile, "w")
sqlOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_factors;\n".format(Youtfile))
sqlOut.write("insert into joyceho.tensor_models values({0}, {1}, \'{2}\',\'{3}\', {4}, {5}, {6}, {7}, {8});\n".format(exptID, labelID, patientSet, exptDesc, iter, innerIter, ls, ll, 0))
sqlOut.close()

print "Completed Tensor Factorization with ID:{0}".format(exptID)
print "Import sql file " + Ysqlfile
