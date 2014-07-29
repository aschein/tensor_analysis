"""
Command line interface to factorize, project, and predict based on the phenotype

Arguments
--------------
inputFile -  the tensor input file format, where {0} will be used to load tensor and axis
expt -       the experiment id offset (actual exptID will be this + sample)
sample -     the bootstrap sample to evaluate
rank -       number of phenotypes to learn for the tensor factorization
iter -       the maximum number of outer iterations
desc -       experiment description for the database table
patientSet - description of the patients in the tensor
testSize -   optional parameter specifying the size of the test population, defaults to 0.2
bootstrap -  optional parameter for the number of total bootstrap samples, defaults to 10.
             note this needs to be > sample
"""
import argparse
import numpy as np
import shelve
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import NMF

import sptensor
import sptenmat
import KLprojection
import decompTools
import predictionModel
import CP_APR

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
totSamples = args.bootstrap
testSize = args.testSize
exptDesc = args.desc
patientSet = args.patientSet
seed = 10
labelID = -1
innerIter = 10
outerIter = args.iter
R = args.rank

X = sptensor.loadTensor(inputFile.format("data"))
yaxis = decompTools.loadAxisInfo(inputFile.format("info"))
tensorInfo = shelve.open(inputFile.format("info"), "r")
Y = np.array(tensorInfo["class"], dtype='int')
tensorInfo.close()

factorFile = "results/apr-raw-{0}.dat".format(exptID)
Youtfile = "results/apr-db-{0}-{1}.csv".format(exptID, args.iter)
Ysqlfile = "results/apr-sql-{0}.sql".format(exptID)
coefFile = "results/logr-coef-{0}.csv".format(exptID)

ttss = StratifiedShuffleSplit(Y,n_iter=totSamples, test_size=testSize, random_state=seed)
print "Starting Tensor Prediction with ID:{0}".format(exptID)
predModel = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
n = 0

def nmfTransform(nmfModel, flatX):
    """ Replace the existing numpy implementation to work on sparse tensor """
    W = np.zeros((flatX.shape[0], nmfModel.n_components_))
    for j in xrange(0, flatX.shape[0]):
        W[j, :], _ = nnls(nmfModel.components_.T, np.ravel(flatX.getrow(j).todense()))
    return W

def getFlatFeature(idx, axisList):
    diagIdx = idx / 470
    medIdx = idx % 470
    return axisList[1][diagIdx] + axisList[2][medIdx]
    
for train, test in ttss:
    if n != nSample:
        n = n + 1
        continue
    else:
        trainShape = list(X.shape)
        train[0] = len(train)
        trainX = predictionModel.tensorSubset(X, train, trainShape)
        
        ## Do the tensor factorization
        np.random.seed(seed)
        M, cpstats, mstats = CP_APR.cp_apr(trainX, R, maxiters=outerIter, maxinner=10)
        M.normalize_sort(1)
        M.writeRawFile(factorFile)
        Yout = decompTools.getDBOutput(M, yaxis)
        Yout = np.column_stack((np.repeat(exptID, Yout.shape[0]), Yout))
        np.savetxt(Youtfile, Yout, fmt="%s", delimiter="|")
        sqlOut = file(Ysqlfile, "w")
        sqlOut.write("load data local infile '/home/joyce/workspace/Health/analysis/tensor/{0}' into table tensor_factors fields terminated by '|'  ;\n".format(Youtfile))
        sqlOut.write("insert into tensor_models(expt_ID, label_ID, description, rank, iterations, inner_iterations, seed, least_squares, log_likelihood, kkt_violation) values({0}, {1}, \'{2}\', {3}, {4}, {5}, {6}, {7}, {8}, {9});\n".format(exptID, labelID, exptDesc, R, outerIter, innerIter, seed, mstats['LS'], mstats['LL'], mstats['KKT']))

        klp = KLProjection.KLProjection(M.U, M.R)
        ptfFeat = klp.projectSlice(X, 0)
        trainY = Y[train]
        predModel.fit(ptfFeat[train, :], trainY)
        ptfPred = predModel.predict_proba(ptfFeat[test,:])
        fpr, tpr, thresholds = metrics.roc_curve(Y[test], ptfPred[:, 1], pos_label=1)
        auc = metrics.auc(fpr, tpr)
        
        coefOut = np.column_stack((np.repeat(exptID, R), np.repeat(nSample, R), np.repeat("l1 Logistic", R), np.array(np.repeat("LogR Coefficient", R), dtype="S100"), np.array(np.repeat("CP_APR", R), dtype="S100"), np.arange(R), predModel.coef_.flatten()))
        np.savetxt(coefFile, coefOut, fmt="%s", delimiter="|")
        sqlOut.write("load data local infile '/home/joyce/workspace/Health/analysis/tensor/{0}' into table predictive_metrics fields terminated by '|'  ;\n".format(coefFile))
        sqlOut.write("insert into predictive_metrics(expt_ID, sample, feature, model, factor, value) values({0}, {1}, \'{2}\',\'{3}\', NULL, {4});\n".format(exptID, nSample, "AUC", "CP_APR", auc))

        ## now we want to do NMF as well
        flatX =  sptenmat.sptenmat(X, [0]).tocsrmat() # matricize along the first mode
        nmfModel = NMF(n_components=R, max_iter = outerIter, nls_max_iter = innerIter, random_state=seed)
        nmfModel.fit(flatX[train, :])
        nmfFeat = nmfTransform()
        
#         nmfVals = nmfModel.components_
#         nmfLambda = np.sum(nmfVals, axis=1)
#         normVals = nmfVals
#         for i in range(R):
#             normVals[i, :] = normVals[i,:] / nmfLambda[i]
#         ## sort by max
#         sortidx = np.argsort(nmfLambda)[::-1];
#         nmfLambda = nmfLambda[sortidx];
#         normVals = normVals[sortidx, :];
#         
#         nmfOut = np.zeros(3)
#         
#         for i in range(R):
#             idx = np.flatnonzero(nmfVals[i,:])
#             nmfOut = np.vstack((nmfOut, np.column_stack((idx, np.repeat(i, len(idx)), nmfVals[i,idx]))))
#             
#         nmfFeat = nmfTransform()
        
        sqlOut.close()
        print "Load SQL FILE " + Ysqlfile + "\n"
        break