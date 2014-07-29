"""
Command line interface to factorize, project, and predict based on the phenotype getting data
for the predictive model table

Arguments
--------------
inputFile -  the tensor input file format, where {0} will be used to load tensor and axis
expt -       the experiment id
patientSet - description of the patients in the tensor
rank -       number of phenotypes to learn for the tensor factorization
iter -       the maximum number of outer iterations
desc -       experiment description for the database table
testSize -   optional parameter specifying the size of the test population, defaults to 0.2
"""
import sptensor
import shelve
import predictionModel
import numpy as np
import argparse

#### Pass in the rank from argparse
parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="input file format")
parser.add_argument("exptID", type=int, help="experiment ID")
parser.add_argument("patientSet", help="Patient set description")
parser.add_argument("rank", type=int, help="rank of the decomposition")
parser.add_argument("iter", type=int, help="the number of iterations")
parser.add_argument("-t", "--testSize", type=float, help="test size", default=0.5)
args = parser.parse_args()

rank = args.rank
iter = args.iter
inputFile = args.inputFile
exptID = args.exptID
patientSet = args.patientSet
outsql = "results/pred-model-{0}-{1}.sql".format(exptID, rank)

print "Using Rank {0} and iterations {1} and test size {2}".format(rank, iter, args.testSize)

## Load information to run the tests
X = sptensor.loadTensor(inputFile.format("data"))
tensorInfo = shelve.open(inputFile.format("info"), "r")
Y = tensorInfo["class"]
XAxis = tensorInfo["axis"]
tensorInfo.close()
Y = np.array(Y, dtype=int)
pm = predictionModel.predictionModel(X, XAxis, Y, rank, testSize=args.testSize, outerIter=iter)
output = pm.evaluatePrediction()
output = np.column_stack((np.repeat(exptID, output.shape[0]), output))
outputFile = "results/pred-model-{0}-{1}.csv".format(exptID, rank)
np.savetxt(outputFile, output, delimiter=",")

sqlOut = file(outsql, "w")
sqlOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel, insert into joyceho.predictive_results;\n".format(outputFile))
sqlOut.write("insert into joyceho.predictive_models values({0}, {1}, \'{2}\',{3}, {4});\n".format(exptID, rank, patientSet, iter, 10))
sqlOut.close()
