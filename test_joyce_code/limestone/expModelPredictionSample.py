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
from scipy.optimize import nnls
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import RandomizedPCA
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import nimfa
import itertools

import sys
sys.path.append("..")

import decompTools
import sptensor
import sptenmat
import KLProjection
import predictionModel
import CP_APR
import khatrirao

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="input file")
parser.add_argument("expt", type=int, help="experiment id offset")
parser.add_argument("sample", type=int, help="the sample to evaluate")
parser.add_argument("rank", type=int, help="rank to evaluate")
parser.add_argument("iter", type=int, help="the number of iterations")
parser.add_argument("-t", "--testSize", type=float, help="test size", default=0.2)
parser.add_argument("-n", "--bootstrap", type=int, help="number of bootstrap samples", default=10)
args = parser.parse_args()

inputFile = args.inputFile
nSample = args.sample
exptID = args.expt + nSample
totSamples = args.bootstrap
testSize = args.testSize
seed = 10
innerIter = 10
outerIter = args.iter
R = args.rank
zeroThr = 1e-4

X = sptensor.loadTensor(inputFile.format("data"))
yaxis = decompTools.loadAxisInfo(inputFile.format("info"))
tensorInfo = shelve.open(inputFile.format("info"), "r")
Y = np.array(tensorInfo["class"], dtype='int')
tensorInfo.close()

diagMed = [[a, b] for a, b in itertools.product(yaxis[1], yaxis[2])] 

predFile = "results/pred-metric-{0}-{1}.csv".format(exptID, nSample)

ttss = StratifiedShuffleSplit(Y,n_iter=totSamples, test_size=testSize, random_state=seed)
print "Starting Tensor Prediction with ID:{0}".format(exptID)
n = 0

def getAUC(model, feat, Y, train, test):
    trainY = Y[train]
    model.fit(feat[train, :], trainY)
    modelPred = model.predict_proba(feat[test,:])
    fpr, tpr, thresholds = metrics.roc_curve(Y[test], modelPred[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def getFlatFeature(idx, axisList):
    diagIdx = idx / 470
    medIdx = idx % 470
    return axisList[1][diagIdx] + axisList[2][medIdx]

def nmfTransform(R, nmfResult, flatX):
    """ Replace the existing numpy implementation to work on sparse tensor """
    W = np.zeros((flatX.shape[0], R))
    coef = nmfResult.coef().todense().transpose()
    for j in xrange(0, flatX.shape[0]):
        W[j, :], _ = nnls(coef, np.ravel(flatX.getrow(j).todense()))
    return W

def getDBEntry(featureName, m):
    output = np.zeros((1, 4))
    for r in range(R):
        # get the nonzero indices
        idx = np.flatnonzero(m[:, r])
        tmp = np.column_stack((np.array(diagMed)[idx], np.repeat(r, len(idx)), m[idx, r]))
        output = np.vstack((output, tmp))
    output = np.delete(output, (0), axis=0)
    output = np.column_stack((np.repeat(exptID, output.shape[0]), np.repeat(featureName, output.shape[0]), output))
    return output

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
        # zero out the small factors
        for n in range(1,2):
            zeroIdx = np.where(M.U[n] < zeroThr)
            M.U[n][zeroIdx] = 0
        klp = KLProjection.KLProjection(M.U, M.R)
        ptfFeat = klp.projectSlice(X, 0)
        ptfMatrix = khatrirao.khatrirao(M.U[1], M.U[2])
        dbOutput = getDBEntry("CP-APR", ptfMatrix)
        
        ## now we want to do PCA and NMF as well
        flatX =  sptenmat.sptenmat(X, [0]).tocsrmat() # matricize along the first mode
        pcaModel = RandomizedPCA(n_components=R)
        pcaModel.fit(flatX[train, :])
        pcaFeat = pcaModel.transform(flatX)
        pcaBasis = pcaModel.components_
        dbOutput = np.vstack((dbOutput, getDBEntry("PCA", pcaBasis)))
        
        nmfModel = nimfa.mf(flatX[train,:], method="nmf", max_iter=outerIter, rank=R)
        nmfResult = nimfa.mf_run(nmfModel)
        nmfFeat = nmfTransform(R, nmfResult, flatX)
        ## get the basis to be stored off
        nmfBasis = nmfResult.coef().transpose()
        nmfBasis = preprocessing.normalize(nmfBasis, norm="l1", axis=0)
        nmfBasis = nmfBasis.toarray()
        zeroIdx = np.where(nmfBasis < zeroThr*zeroThr)
        nmfBasis[zeroIdx]= 0
        dbOutput = np.vstack((dbOutput, getDBEntry("NMF", nmfBasis)))

        ## write the DBOutput
        Youtfile = "results/pred-db-{0}.csv".format(exptID)
        np.savetxt(Youtfile, dbOutput, fmt="%s", delimiter="|")
        
        # ## do the list of models
        # logModel = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        # svmModel = svm.SVC()
        
        # predOut = np.array([exptID, "L1 Logistic", "AUC", "CP_APR", "\N", getAUC(logModel, ptfFeat, Y, train, test)])
        # predOut = np.vstack((predOut, np.array([exptID, "L1 Logistic", "AUC", "PCA", "\N", getAUC(logModel, pcaFeat, Y, train, test)])))
        # predOut = np.vstack((predOut, np.array([exptID, "L1 Logistic", "AUC", "NMF", "\N", getAUC(logModel, nmfFeat, Y, train, test)])))
        # predOut = np.vstack((predOut, np.array([exptID, "SVM", "AUC", "CP_APR", "\N", getAUC(svmModel, ptfFeat, Y, train, test)])))
        # predOut = np.vstack((predOut, np.array([exptID, "SVM", "AUC", "PCA", "\N", getAUC(svmModel, pcaFeat, Y, train, test)])))
        # predOut = np.vstack((predOut, np.array([exptID, "SVM", "AUC", "NMF", "\N", getAUC(svmModel, nmfFeat, Y, train, test)])))
        # np.savetxt(predFile, predOut, fmt="%s", delimiter="|")
        
        # Ysqlfile = "results/pred-sql-{0}.sql".format(exptID)
        # sqlOut = file(Ysqlfile, "w")
        # sqlOut.write("load data local infile '/home/joyce/workspace/Health/analysis/tensor/{0}' into table predictive_factors fields terminated by '|'  ;\n".format(Youtfile))
        # sqlOut.write("load data local infile '/home/joyce/workspace/Health/analysis/tensor/{0}' into table predictive_metrics fields terminated by '|'  ;\n".format(predFile))
        # sqlOut.write("insert into predictive_models(expt_ID, rank, iterations, seed) values({0}, {1}, {2}, {3});\n".format(exptID, R, outerIter, seed))
        # sqlOut.close()

        break