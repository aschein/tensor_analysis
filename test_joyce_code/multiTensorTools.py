""" 
File that contains common methods used during multi-tensor factorization
"""
import itertools
import numpy as np
import shelve

import sptensor

def saveMultiTensor(X, sharedModes, axisDict, patClass, outFilename):
    """  
    Save this multi-way tensor (the original data + axis information)
    The "data" file contains the raw tensor information in numpy binary format
    First is the number of tensors in the list, then the sptensor information, and finally shared modes
    
    Parameters
    ------------
    X : a list of tensors to save
    sharedModes: a 2-d numpy array specifying common shared modes
    axisDict : a mapping between the indices and the actual axis values
    patClass : a map between patients and the labels
    outFilename : the pattern for the output format, note that {0} is necessary as 2 files are produced
    """
    outfile = file(outFilename.format("data"), "wb")
    np.save(outfile, len(X))
    for tx in X:
        np.save(outfile, tx.subs)
        np.save(outfile, tx.vals)
        np.save(outfile, tx.shape)
    
    np.save(outfile, sharedModes)
    outfile.close()
    
    ## store the information into a datafile
    tensorInfo = shelve.open(outFilename.format("info"), "c")
    tensorInfo['axis'] = axisDict
    tensorInfo['class'] = patClass
    tensorInfo.close()
    
def loadMultiTensor(inFilename):
    """ 
    Load the list of tensors from this input file format
    
    Parameters
    ------------
    inFilename : the input file pattern for the 2 files with the tensor data and axis information
    
    Output
    -----------
    X : the list of tensors in the file
    sharedModes : the 2-d array with the shared modes location
    axisDict : the axis information for all the tensors
    patClass : the patient cohort information
    """
    infile = file(inFilename.format("data"), "rb")
    lenX = np.load(infile)
    X = []
    for i in range(lenX):
        subs = np.load(infile)
        vals = np.load(infile)
        siz = np.load(infile)
        X.append(sptensor.sptensor(subs, vals, siz))
    sharedModes = np.load(infile)
    tensorInfo = shelve.open(inFilename.format("info"), "r")
    axisDict = tensorInfo['axis']
    patClass = tensorInfo['class']
    tensorInfo.close()
    
    return X, sharedModes, axisDict, patClass

def hardThreshold(factX, zeroThr=1e-4, modeZeroThr=1e-20):    
    """ 
    Perform the hard thresholding on the factorization
    
    Parameters
    ------------
    factX : the list of tensor factorizations
    zeroThr : the threshold parameter
    modeZeroThr: the value to threshold the zeroth mode
    
    Output
    -----------
    factX : the list of factorizations with the thresholding operation performed
    """
    for i in range(len(factX)):
        for n in range(1, factX[i].ndims()):
            zeroIdx = np.where(factX[i].U[n] < zeroThr)
            factX[i].U[n][zeroIdx] = 0
        zeroIdx = np.where(factX[i].U[0] < modeZeroThr)
        factX[i].U[0][zeroIdx] = 0
    return factX

def saveRawFile(factX, outFilename):    
    """ 
    Save the tensor factorization in the output file
    
    Parameters
    ------------
    factX : the list of factorized tensors
    outFilename : the output file name
    """
    outfile = file(outFilename, "wb")
    np.save(outfile, len(factX))
    for kt in factX:
        np.save(outfile, kt.lmbda)
        np.save(outfile, kt.ndims())
        for n in range(kt.ndims()):
            np.save(outfile, kt.U[n])
    outfile.close()
    
def getDBFormat(factX, modes, axisDict):
    """ 
    Get the database format for this set of factorizations
    
    Parameters
    ------------
    factX : the list of tensor factorizations
    modes: a list of all the unique modes in the tensor list
    axisDict : the mapping between the tensor modes and the axis information
    
    Output
    -----------
    outStack: a 2-D array with 4 columns with mode, feature_name (derived from axisDict), factor, and value
    """
    outStack = np.zeros((1, 4))
    ## Get the lambdas
    for i in range(len(factX)):
        idx = np.flatnonzero(factX[i].lmbda)
        tmp = np.column_stack((np.repeat(-1 * (i+1), len(idx)), np.repeat("lambda", len(idx)).astype('S100'), idx, factX[i].lmbda[idx]))
        outStack = np.vstack((outStack, tmp))
    
    midx = 0
    for m in modes:
        tensorIdx = m.flatten()[0]
        tensorDim = m.flatten()[1]
        tTuple = (tensorIdx, tensorDim)
        print tTuple
        for r in range(factX[tensorIdx].R):
            idx = np.flatnonzero(factX[tensorIdx].U[tensorDim][:,r])
            tmp = np.column_stack((np.repeat(midx, len(idx)), np.array(axisDict[tTuple], dtype="S100")[idx], np.repeat(r, len(idx)), factX[tensorIdx].U[tensorDim][idx, r]))
            outStack = np.vstack((outStack, tmp))
        midx = midx + 1
    
    return outStack

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

def tensorSubset(X, sm, subsetIds):
    """ 
    Get a subset of the tensors specified by the subsetIds
    
    Parameters
    ------------
    X : a list of tensors to subset
    sm : a 2-d numpy array specifying the tensor mode locations to compute the subset on
    subsetIds : a list of indices
    
    Output
    -----------
    subsetX : a list of tensors with the indices rebased
    """
    subsetX = [ti for ti in X]
    for row in range(sm.shape[0]):
        tensorIdx = sm[row, 0]
        tensorMode = sm[row, 1]
        subsetIdx = np.in1d(X[tensorIdx].subs[:,tensorMode].ravel(), subsetIds)
        subsIdx = np.where(subsetIdx)[0]
        subsetSubs = X[tensorIdx].subs[subsIdx,:]
        subsetVals = X[tensorIdx].vals[subsIdx]
        subsetSubs = rebase(subsetIds, subsetSubs)
        subsetShape = list(X[tensorIdx].shape)
        subsetShape[tensorMode] = len(subsetIds)
        subsetX[tensorIdx] = sptensor.sptensor(subsetSubs, subsetVals, subsetShape)
    return subsetX