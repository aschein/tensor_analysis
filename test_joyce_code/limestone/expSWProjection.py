from scipy.sparse import coo_matrix
import numpy as np
import argparse

import CP_APR
import sptensor
import ktensor
import KLProjection
import decompTools

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="input file to parse")
parser.add_argument("expt", type=int, help="experiment number")
parser.add_argument("description", help="description of experiment")
parser.add_argument("swInc", type=int, help="sliding window increment")
parser.add_argument("startInc", type=int, help="sliding window start")
parser.add_argument("endInc", type=int, help="sliding window start")
parser.add_argument("-l", "--label", type=int, help="label ID for patient set", default=-1)
parser.add_argument("-i", "--iterations", type=int, help="Number of outer interations", default=70)
parser.add_argument("-r", "--rank", type=int, help="Decomposition Rank", default=100)
args = parser.parse_args()

exptID = args.expt
inputFile = args.inputFile
labelID = args.label
rank = args.rank
outerIter = args.iterations
Moutfile = "results/apr-db-{0}-{1}.csv".format(exptID, iter)
Msqlfile = "results/sw-sql-{0}.sql".format(exptID)
MrawFile = "results/sw-raw-{0}.dat".format(exptID)
SWoutfile = "results/sw-db-{0}.dat".format(exptID)
timeRange = range(args.startInc, args.endInc, args.swInc)

def findFactors(X, R=100, outerIter=70, innerIter=10, zeroThr=1e-4):
    """ Find the factor basis for this tensor """
    M, cpstats, mstats = CP_APR.cp_apr(X, R=R, maxiters=outerIter, maxinner=innerIter)
    M.normalize_sort(1)
    M = decompTools.zeroSmallFactors(M, zeroThr)
    return KLProjection.KLProjection(M.U, R), M, mstats

def getDBRepresentation(pf, axis):
    """ 
    Write output for database table sliding window
    
    Parameters
    -----------------------
    PF : the factor matrix where rows is patients and the column are factor values
    axis : the axis label of patients PIDs
    
    """
    factors = pf.shape[1] # the number of columns
    rows = pf.shape[0]
    idx = np.flatnonzero(pf[:, 0])
    dbOut = np.column_stack((axis[idx], np.repeat(0, len(idx)), pf[idx,0]))
    for col in range(1, factors):
        idx = np.flatnonzero(pf[:, col])
        dbOut = np.vstack((dbOut, np.column_stack((axis[idx], np.repeat(col, len(idx)), pf[idx,col]))))
    return dbOut

refX = sptensor.loadTensor(inputFile.format(0, "data"))
refAxis = decompTools.loadAxisInfo(inputFile.format(0, "info"))
## Find the factors for the first one
klp, M, mstats = findFactors(refX, R=rank, outerIter=outerIter, innerIter=10)

## Store off the factors to be loaded into a database
M.writeRawFile(MrawFile)
Mout = decompTools.getDBOutput(M, refAxis)
Mout = np.column_stack((np.repeat(exptID, Mout.shape[0]), Mout))
np.savetxt(Moutfile, Yout, fmt="%s", delimiter="|")

sqlOut = file(Ysqlfile, "w")
## write the factors and the models into the database
sqlOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_factors;\n".format(Youtfile))
sqlOut.write("insert into joyceho.tensor_models values({0}, {1}, \'{2}\',\'{3}\', {4}, {5}, {6}, {7}, {8});\n".format(exptID, labelID, patientSet, exptDesc, iter, innerIter, mstats['LS'], mstats['LL'], mstats['KKT']))

output = np.zeros((1, 4)) 
for i in timeRange:
    X = sptensor.loadTensor(inputFile.format(i, "data"))
    XAxis = decompTools.loadAxisInfo(inputFile.format(i, "info"))
    # Project the sliding window onto the factor
    patFactor = klp.projectSlice(X, 0)
    pf = getDBRepresentation(patFactor, np.array(XAxis[0], dtype="int"))
    output = np.vstack((output, np.column_stack((np.repeat(i, pf.shape[0]), pf))))

output = np.delete(output, (0), axis=0)
np.savetxt(SWoutfile, output, delimiter="|")
sqlOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.sliding_window;\n".format(SWoutfile))
sqlOut.close()
