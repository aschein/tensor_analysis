import itertools
import numpy as np
from sklearn import metrics

import sptensor

def getAUC(model, feat, Y, train, test):
    trainY = Y[train]
    model.fit(feat[train, :], trainY)
    modelPred = model.predict_proba(feat[test,:])
    fpr, tpr, thresholds = metrics.roc_curve(Y[test], modelPred[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc, modelPred

def createRawFeatures(X):
    mode2Offset = X.shape[1]
    rawFeat = np.zeros((X.shape[0], mode2Offset+X.shape[2]))
    for k in range(X.subs.shape[0]):
        sub = X.subs[k,:]
        rawFeat[sub[0], sub[1]] = rawFeat[sub[0], sub[1]] + X.vals[k,0]
        rawFeat[sub[0], mode2Offset + sub[2]] = rawFeat[sub[0], mode2Offset + sub[2]] + X.vals[k,0]
    return rawFeat

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
    subsetSubs = rebase(subsetIds, subsetSubs)
    return sptensor.sptensor(subsetSubs, subsetVals, subsetShape)