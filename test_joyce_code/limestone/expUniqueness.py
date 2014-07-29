"""
Command line interface to compare uniqueness of factorizations

Arguments
--------------
inputFile -    tensor input file format with {0} for the 2 separate files
expt -         the experiment number for these sets of parameters
description -  the experiment description for the database table
label -        the patient cohort (all case, control or mixed)
interations -  the maximum number of outer iterations defaulting to 100
starts -       the number of random initializations defaulting to 10
rank -         the decomposition rank for the factorization
"""
import decompTools
import sptensor
import ktensor
import CP_APR
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="input file to parse")
parser.add_argument("expt", type=int, help="experiment number")
parser.add_argument("description", help="description of experiment")
parser.add_argument("label", type=int, help="label ID for patient set")
parser.add_argument("-i", "--iterations", type=int, help="Number of outer iterations", default=100)
parser.add_argument("-n", "--starts", type=int, help="Number of random starts", default=10)
parser.add_argument("-r", "--rank", type=int, help="Rank of factorization", default=10)
args = parser.parse_args()

####### EXPERIMENT SECTION ############
exptID = args.expt
labelID = args.label
exptDesc = args.description
totalIter = args.starts
R = args.rank
maxIters = args.iterations
innerIters = 10
seedArray = np.arange(0, 10*totalIter, 10)

inputFile = args.inputFile.format("data")
yaxis = decompTools.loadAxisInfo(args.inputFile.format("info"))

outfilePattern = 'results/unique-db-{0}-{1}.csv'
rawfilePattern = 'results/unique-raw-{0}-{1}.dat'
scorefilePattern = 'results/unique-scores-{0}.csv'
sqlOutfile = "results/unique-sql-{0}.sql".format(exptID)

yFactor = []

print "Running Uniqueness Experiment with ID {0} and iterations {1}".format(exptID, maxIters)
modelOut = file(sqlOutfile, "w")

for i in range(totalIter):
    # initialize the seed for repeatability
    np.random.seed(seedArray[i])
    print "Random Start with seed {0}".format(seedArray[i])
    Y, ystats, mstats = decompTools.decomposeCountTensor(inputFile, R=R, outerIters=maxIters, innerIters=innerIters, zeroTol=1e-4)
    Y.writeRawFile(rawfilePattern.format(exptID,i))
    dbYFile = outfilePattern.format(exptID, i)
    dbOut = decompTools.getDBOutput(Y, yaxis)
    dbOut = np.column_stack((np.repeat(exptID, dbOut.shape[0]), np.repeat(i, dbOut.shape[0]), dbOut))
    dbOut = np.insert(dbOut, 4, np.repeat(-100, dbOut.shape[0]), axis=1)
    np.savetxt(dbYFile, dbOut, fmt="%s", delimiter="|")
    yFactor.append(ktensor.ktensor(Y.lmbda.copy(), [Y.U[n].copy() for n in range(Y.ndims())]))
    # write to the sequel file for ease
    modelOut.write("insert into joyceho.tensor_uniq_models values({0},{1},{2},\'{3}\',{4},{5},{6},{7},{8});\n".format(exptID, i, labelID, exptDesc, maxIters, innerIters, mstats['LS'], mstats['LL'], mstats['KKT']))
    modelOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_uniq_results;\n".format(dbYFile))

## Calculate all the scores
def __generateInfo(n, exptID, type, method, i, k):
    info = np.tile(np.array([exptID, type, method, i, k], dtype="S20"), n)
    info = info.reshape((n, 5))
    return info

scoreResults = np.empty((1,9), dtype="S20")
for i in range(totalIter):
    for k in range(i+1, totalIter):
        A = yFactor[i]
        B = yFactor[k]
        # Calculate FMS values
        topFMS = np.around(A.top_fms(B), 10)
        info = __generateInfo(topFMS.shape[0], exptID, "fms", "top10", i, k)
        scoreResults = np.append(scoreResults, np.column_stack((info, topFMS)), axis=0)
        greedyFMS = np.around(A.greedy_fms(B))
        info = __generateInfo(greedyFMS.shape[0], exptID, "fms", "greedy", i, k)
        scoreResults = np.append(scoreResults, np.column_stack((info, greedyFMS)), axis=0)
        # Calculate FOS values
        topFOS = np.around(A.top_fos(B), 10)
        info = __generateInfo(topFOS.shape[0], exptID, "fos", "top10", i, k)
        scoreResults = np.append(scoreResults, np.column_stack((info, topFOS)), axis=0)
        greedyFOS = np.around(A.greedy_fos(B))
        info = __generateInfo(greedyFOS.shape[0], exptID, "fos", "greedy", i, k)
        scoreResults = np.append(scoreResults, np.column_stack((info, greedyFOS)), axis=0)

scoreResults = np.delete(scoreResults, (0), axis=0)
np.savetxt(scorefilePattern.format(exptID), scoreResults, fmt="%s", delimiter="|")
modelOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_uniq_scores;\n".format(scorefilePattern.format(exptID)))
modelOut.close()

print "Complete Uniqueness Experiment with ID:{0}".format(exptID)