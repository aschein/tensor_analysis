"""
Simultaneous tensor factorization prediction experiment

Arguments
--------------
inputFile -    tensor input file format with {0} for the 2 separate files
expt -         the experiment number for these sets of parameters
sample -       the bootstrap sample to evaluate
rank -         the decomposition rank for the factorization
iterations -   maximum number of outer iterations 
description -  the experiment description for the database table
patientSet -   the patient set description for the database table
testSize -     optional parameter to specify the size of the train/test split, defaults to 0.2
bootstrap -    optional the total number of bootstrap samples, defaults to 10
"""
import argparse
import numpy as np
import shelve
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import NMF
from pymongo import MongoClient

import sys
sys.path.append("..")

import tensorTools
import sim_APR
import simKLProjection

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="input file")
parser.add_argument("expt", type=int, help="experiment id offset")
parser.add_argument("sample", type=int, help="the sample to evaluate")
parser.add_argument("rank", type=int, help="rank to evaluate")
parser.add_argument("iter", type=int, help="the number of iterations")
parser.add_argument("desc", help="experiment description")
parser.add_argument("patientSet", help="patient set")
parser.add_argument("-t", "--testSize", type=float, help="test size", default=0.2)
parser.add_argument("-n", "--bootstrap", type=int, help="number of bootstrap samples", default=10)
args = parser.parse_args()

inputFile = args.inputFile
nSample = args.sample
exptID = args.expt + nSample
seed = 10
totSamples = args.bootstrap
testSize = args.testSize
labelID = -1
exptDesc = args.desc
patientSet = args.patientSet
innerIter = 10
outerIter = args.iter
R = args.rank

X, sharedModes, axisDict, patClass = multiTensorTools.loadMultiTensor(inputFile)
Y = np.array(patClass.values(), dtype="int")
ttss = StratifiedShuffleSplit(Y ,n_iter=totSamples, test_size=testSize, random_state=seed)
predModel = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

factorFile = "results/multi-raw-{0}.dat".format(exptID)
Youtfile = "results/multi-db-{0}-{1}.csv".format(exptID, args.iter)
Ysqlfile = "results/apr-sql-{0}.sql".format(exptID)
coefFile = "results/logr-coef-{0}.csv".format(exptID)

print "Starting Tensor Prediction with ID:{0}".format(exptID)
n = 0

for train, test in ttss:
    if n != nSample:
        n = n + 1
        continue
    else:
        trainX = multiTensorTools.tensorSubset(X, sharedModes[0], train)
        sapr = sim_APR.SAPR(trainX, R, sharedModes, maxiters=outerIter, maxinner = innerIter)
        M, cpstats, mstats = sapr.factorize()
        multiTensorTools.saveRawFile(M, factorFile)
        ## hard threshold the small factors
        M = multiTensorTools.hardThreshold(M, 1e-4)
        ## save the files
        Yout = multiTensorTools.getDBFormat(sapr.M, sapr.modes, axisDict)
        Yout = np.column_stack((np.repeat(exptID, Yout.shape[0]), Yout))
        np.savetxt(Youtfile, Yout, fmt="%s", delimiter="|")
        ## write the sql file
        sqlOut = file(Ysqlfile, "w")
        sqlOut.write("load data local infile '/home/joyce/workspace/Health/analysis/tensor/{0}' into table tensor_factors fields terminated by '|'  ;\n".format(Youtfile))
        sqlOut.write("insert into tensor_models(expt_ID, label_ID, description, rank, iterations, inner_iterations, seed, least_squares, log_likelihood, kkt_violation) values({0}, {1}, \'{2}\', {3}, {4}, {5}, {6}, {7}, {8}, {9});\n".format(exptID, labelID, exptDesc, R, outerIter, innerIter, seed, mstats['LS'], mstats['LL'], mstats['KKT']))

        klp = simKLProjection.simKLProjection(M, R)
        ptfFeat = klp.projectSlice(X, sharedModes[0])
        trainY = Y[train]
        predModel.fit(ptfFeat[train, :], trainY)
        ptfPred = predModel.predict_proba(ptfFeat[test,:])
        fpr, tpr, thresholds = metrics.roc_curve(Y[test], ptfPred[:, 1], pos_label=1)
        auc = metrics.auc(fpr, tpr)
        coefOut = np.column_stack((np.repeat(exptID, R), np.repeat(nSample, R), np.array(np.repeat("LogR Coefficient", R), dtype="S100"), np.array(np.repeat("SIM_APR", R), dtype="S100"), np.arange(R), predModel.coef_.flatten()))
        np.savetxt(coefFile, coefOut, fmt="%s", delimiter="|")
        sqlOut.write("load client from /home/joyceho/workspace/tensor/{0} of del modified by coldel| insert into joyceho.predictive_metrics;\n".format(coefFile))
        sqlOut.write("insert into joyceho.predictive_metrics values({0}, {1}, \'{2}\',\'{3}\', NULL, {4});\n".format(exptID, nSample, "AUC", "SIM_APR", auc))

        sqlOut.close()
        print "Load SQL FILE " + Ysqlfile + "\n"
        break