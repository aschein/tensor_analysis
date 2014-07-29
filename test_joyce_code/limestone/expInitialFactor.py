import numpy as np
import shelve
import sys
sys.path.append("..")

import sptensor
import ktensor
import CP_APR
import pmdTools

def useHier(topX, regX, R, hierIters, hierInner, regIters, regInner, tensorInfo):
    topY1, top1stats, top1mstats = CP_APR.cp_apr(topX, R, maxiters=hierIters, maxinner=hierInner)
    # reduce them to probability and then just sort them
    topY1.normalize_sort(1)
    topY1 = pmdTools.zeroSmallFactors(topY1, 1e-4)
    ### Use the factors to populate the factors
    Udiag = np.zeros((len(tensorInfo['diag']), R))
    Umed = np.zeros((len(tensorInfo['med']), R))
    ### Patient factors stays the same
    for idx, diag in enumerate(tensorInfo['diag']):
        topDiagIdx = tensorInfo['diagHier'][diag]
        diagCount = tensorInfo['diagHierCount'][topDiagIdx]
        Udiag[idx,:] = topY1.U[1][topDiagIdx,:] / diagCount
    for idx, med in enumerate(tensorInfo['med']):
        topMedIdx = tensorInfo['medHier'][med]
        medCount = tensorInfo['medHierCount'][topMedIdx]        
        Umed[idx,:] = topY1.U[2][topMedIdx,:] / medCount
    Mtop = ktensor.ktensor(np.ones(R), [topY1.U[0].copy(), Udiag, Umed])
    Y1, ystats, mstats = CP_APR.cp_apr(X1, R, Minit=Mtop, maxiters=regIters, maxinner=regInner)
    return Y1, topY1, top1stats, top1mstats, ystats, mstats

def __writeDBFile(X, filename, axisList, modelID, init):
    # for each lambda create the stack
    idx = np.flatnonzero(X.lmbda)
    tempOut = np.column_stack((np.repeat(-1, len(idx)), np.repeat("lambda", len(idx)), np.repeat(-1, len(idx)), idx, X.lmbda[idx]))
    for n in range(X.ndims()):
        for r in range(X.R):
            idx = np.flatnonzero(X.U[n][:, r])
            # get the ones for this mode/factor
            temp = np.column_stack((np.repeat(n, len(idx)), axisList[n][idx], idx, np.repeat(r, len(idx)), X.U[n][idx, r]))
            tempOut = np.vstack((tempOut, temp))
    tempOut = np.column_stack((np.repeat(modelID, tempOut.shape[0]), np.repeat(init, tempOut.shape[0]), tempOut))
    np.savetxt(filename, tempOut, fmt="%s", delimiter="|")
    
##### Experimental section ###############
x1file = "data/hf-label1-hierarchy-{0}.dat"
topX1, X1, tensorInfo = parseFile('data/hf-label1-data.dat', 0, 2, 3, 5, 4, x1file)
labelID = 1
exptID = 604
seedArray = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
R = 100
hierIters = 20 ## Number of higher level iterations
innerIters = 10 ## Number of inner iterations 
regIters = 20
totalIter = 10
expt_desc = "Hierarchy with {0},{1} on HF patients".format(hierIters, regIters)

yTopFactor = []
yTopStats = []
mTopStats = []
yRegFactor = []
yRegStats = []
mRegStats = []

outfile = 'results/unique-hier-db-{0}-{1}.csv'
rawTopFile = 'results/unique-hier-top-raw-{0}-{1}.dat'
rawRegFile = 'results/unique-hier-reg-raw-{0}-{1}.dat'
yaxis = [np.array(tensorInfo['pat']), np.array(tensorInfo['diag']), np.array(tensorInfo['med'])]

modelOut = file("results/uniq-hier-{0}.sql".format(exptID), "w")
for i in range(totalIter):
    # set the seed for repeatability
    np.random.seed(seedArray[i])
    yreg, ytop, ytopstats, ytopMstats, yregstats, yregMstats = useHier(topX1, X1, R, hierIters, innerIters, regIters, innerIters, tensorInfo)
    yreg.writeRawFile(rawRegFile.format(exptID, i))
    ytop.writeRawFile(rawTopFile.format(exptID, i))
    
    yreg.normalize_sort(1)
    yreg = pmdTools.zeroSmallFactors(yreg, 1e-4)
    yfile = outfile.format(exptID, i)
    __writeDBFile(yreg, yfile, yaxis, exptID, i)
    
    mstats = yregMstats
    modelOut.write("insert into joyceho.tensor_uniq_models values({0},{1},{2},\'{3}\',{4},{5},{6},{7},{8});\n".format(exptID, i, labelID, expt_desc, hierIters+regIters, innerIters, mstats['LS'], mstats['LL'], mstats['KKT']))
    modelOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_uniq_results;\n".format(yfile))

    yTopFactor.append(ytop)
    yRegFactor.append(yreg)
    yTopStats.append(ytopstats)
    yRegStats.append(yregstats)
    mTopStats.append(ytopMstats)
    mRegStats.append(yregMstats)

## Calculate all the scores
xfact = 10
fmsResults = np.zeros((1,5))

for i in range(totalIter):
    for k in range(i+1, totalIter):
        A = yRegFactor[i]
        B = yRegFactor[k]
        topScore = A.top_fms(B, xfact)
        greedScore = A.greedy_fms(B)
        fmsResults = np.append(fmsResults, [[exptID, i, k, topScore, greedScore]], axis=0)
        
fmsResults = np.delete(fmsResults, (0), axis=0)
fmsFile = "results/uniq-hier-fms-{0}.csv".format(exptID)
np.savetxt(fmsFile, fmsResults, delimiter="|")
modelOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.tensor_uniq_fms_scores;\n".format(fmsFile))
modelOut.close()

print "Complete Hierarchy Uniqueness Experiment with ID:{0}".format(exptID)