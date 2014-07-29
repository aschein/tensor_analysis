import itertools
import numpy as np
from scipy.optimize import nnls
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression

import ktensor
import sptensor
import sptenmat
import tensor
import CP_APR
import KLProjection
import shelve

def createRawFeatures(X):
    mode2Offset = X.shape[1]
    rawFeat = np.zeros((X.shape[0], mode2Offset+X.shape[2]))
    for k in range(X.subs.shape[0]):
        sub = X.subs[k,:]
        rawFeat[sub[0], sub[1]] = rawFeat[sub[0], sub[1]] + X.vals[k,0]
        rawFeat[sub[0], mode2Offset + sub[2]] = rawFeat[sub[0], mode2Offset + sub[2]] + X.vals[k,0]
    return rawFeat

class predictionModel:
    X = None
    axisInfo = None
    Y = None
    R = 0
    samples = 0
    ttss = None
    innerIter = 10
    outerIter = 70
    rawFeatures = None
    pcaModel = None
    predModel = None
    nmfModel = None
    flatX = None
    
    def __init__(self, X, XAxis, Y, R, outerIter=70, testSize=0.5, samples=10, seed=10):
        self.X = X
        self.axisInfo = np.array(XAxis[0], dtype="int")
        self.Y = Y
        self.R = R
        self.outerIter = outerIter
        self.samples = samples
        self.ttss = StratifiedShuffleSplit(Y,n_iter=samples, test_size=testSize, random_state=seed)
        self.rawFeatures = createRawFeatures(X)
        self.flatX =  sptenmat.sptenmat(X, [0]).tocsrmat() # matricize along the first mode
        self.pcaModel = RandomizedPCA(n_components=R)
        self.predModel = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        self.nmfModel = NMF(n_components=R, max_iter = self.outerIter, nls_max_iter = self.innerIter)
    
    @staticmethod
    def rebase(ids, subs):
        """ 
        Re-index according to the ordered array that specifies the new indices
        
        Parameters
        ------------
        ids : ordered array that embeds the location
        subs : the locations that need to be 'reindexed' according to the ids
        """
        idMap = dict(itertools.izip(ids, range(len(ids))))
        for k in range(subs.shape[0]):
            id = subs[k, 0]
            subs[k, 0] = idMap[id]
        return subs
        
    def findFactors(self, trainX, zeroThr=1e-4):
        """ Find the factor basis for this tensor """
        M, cpstats, mstats = CP_APR.cp_apr(trainX, R=self.R, maxiters=self.outerIter, maxinner=self.innerIter)
        M.normalize_sort(1)
        # zero out the small factors
        for n in range(M.ndims()):
            zeroIdx = np.where(M.U[n] < zeroThr)
            M.U[n][zeroIdx] = 0
        return KLProjection.KLProjection(M.U, self.R)
    
    def nmfTransform(self):
        """ Replace the existing numpy implementation to work on sparse tensor """
        W = np.zeros((self.flatX.shape[0], self.nmfModel.n_components_))
        for j in xrange(0, self.flatX.shape[0]):
            W[j, :], _ = nnls(self.nmfModel.components_.T, np.ravel(self.flatX.getrow(j).todense()))
        return W
        
    def evaluatePrediction(self):
        run = 0
        output = np.zeros((1,7))
        for train, test in self.ttss:
            print "Evaluating Run:{0}".format(run)
            # get the indices for the training tensor
            trainShape = list(self.X.shape)
            trainShape[0] = len(train)
            trainX = tensorSubset(self.X, train, trainShape)
            trainY = self.Y[train]
            ## find the tensor factors for PTF-HT
            klp = self.findFactors(trainX)
            ## Get the reduced features for the data points
            ptfFeat = klp.projectSlice(self.X, 0)
            ## Calculate the PCA baseline
            self.pcaModel.fit(self.flatX[train, :])
            pcaFeat = self.pcaModel.transform(self.flatX)
            ## Calculate the NMF baseline
            self.nmfModel.fit(self.flatX[train, :])
            nmfFeat = self.nmfTransform()
            ## Evaluate the raw fit using logistic regression
            self.predModel.fit(self.rawFeatures[train, :], trainY)
            rawPred = self.predModel.predict_proba(self.rawFeatures[test,:])
            ## Evaluate the PCA fit using logistic regression
            self.predModel.fit(pcaFeat[train, :], trainY)
            pcaPred = self.predModel.predict_proba(pcaFeat[test,:])
            ## Evaluate the baseline features using logistic regression 
            self.predModel.fit(nmfFeat[train, :], trainY)
            basePred = self.predModel.predict_proba(nmfFeat[test,:])
            ## Evaluate the reduced fit using logistic regression
            self.predModel.fit(ptfFeat[train, :], trainY)
            ptfPred = self.predModel.predict_proba(ptfFeat[test,:])
            ## stack the tuples for storage
            testY = self.Y[test]
            temp = np.column_stack((np.repeat(run, len(testY)), self.axisInfo[test], rawPred[:, 1], pcaPred[:,1], basePred[:, 1], ptfPred[:,1], testY))
            output = np.vstack((output, temp))
            run = run + 1
        output = np.delete(output, (0), axis=0)
        return output
    
def tensorSubset(origTensor, subsetIds, subsetShape):
    """ 
    Get a subset of the tensor specified by the subsetIds
    
    Parameters
    ------------
    X : the original tensor
    subsetIds : a list of indices
    subsetShape : the shape of the new tensor
    
    Output
    -----------
    subsetX : the tensor with the indices rebased
    """
    subsetIdx = np.in1d(origTensor.subs[:,0].ravel(), subsetIds)
    subsIdx = np.where(subsetIdx)[0]
    subsetSubs = origTensor.subs[subsIdx,:]
    subsetVals = origTensor.vals[subsIdx]
    # reindex the 0th mode
    subsetSubs = predictionModel.rebase(subsetIds, subsetSubs)
    return sptensor.sptensor(subsetSubs, subsetVals, subsetShape)