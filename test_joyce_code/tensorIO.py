""" 
File that contains common methods for tensor IO
"""
import itertools
import numpy as np
import shelve
import csv

import sptensor

AXIS = "axis"
CLASS = "class"

## Read a file that has 3 modes
def parse3DTensorFile(filename, axis0Dict, axis1Dict, axis2Dict, axis0Idx, axis1Idx, axis2Idx, valueIdx):
	print "Creating tensor from file " + filename
	if axis0Dict is None:
		axis0Dict = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
	if axis1Dict is None:
		axis1Dict = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
	if axis2Dict is None:
		axis2Dict = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
	tensorIdx = np.array([[0, 0, 0]], dtype=int)
	tensorVal = np.array([[0]], dtype=int)
	f = open(filename, "rb")
	for row in csv.reader(f):
		## see if we need to add them to the if
		if not axis0Dict.has_key(row[axis0Idx]):
			axis0Dict[row[axis0Idx]] = len(axis0Dict)
		if not axis1Dict.has_key(row[axis1Idx]):
			axis1Dict[row[axis1Idx]] = len(axis1Dict)
		if not axis2Dict.has_key(row[axis2Idx]):
			axis2Dict[row[axis2Idx]] = len(axis2Dict)
		axis0Id = axis0Dict.get(row[axis0Idx])
		axis1Id = axis1Dict.get(row[axis1Idx])
		axis2Id = axis2Dict.get(row[axis2Idx])
		tensorIdx = np.vstack((tensorIdx, [[axis0Id, axis1Id, axis2Id]]))
		tensorVal = np.vstack((tensorVal, [[int(row[valueIdx])]]))
	tensorIdx = np.delete(tensorIdx, (0), axis=0)
	tensorVal = np.delete(tensorVal, (0), axis=0)
	f.close()
	tenX = sptensor.sptensor(tensorIdx, tensorVal, np.array([len(axis0Dict), len(axis1Dict), len(axis2Dict)]))
	axisDict = {0: axis0Dict, 1: axis1Dict, 2: axis2Dict}
	return tenX, axisDict

def parseShared2DTensorFile(filename, axis0Dict, axis1Dict, axis0Idx, axis1Idx, valueIdx):
	print "Creating tensor from file " + filename
	## initialize the dictionaries if nonexistent
	if axis0Dict is None:
		axis0Dict = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
	if axis1Dict is None:
		axis1Dict = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
	tensorIdx = np.array([[0, 0]], dtype=int)
	tensorVal = np.array([[0]], dtype=int)
	f = open(filename, "rb")
	for row in csv.reader(f):
		## see if we need to add them to the if
		if not axis0Dict.has_key(row[axis0Idx]):
			axis0Dict[row[axis0Idx]] = len(axis0Dict)
		if not axis1Dict.has_key(row[axis1Idx]):
			axis1Dict[row[axis1Idx]] = len(axis1Dict)
		axis0Id = axis0Dict.get(row[axis0Idx])
		axis1Id = axis1Dict.get(row[axis1Idx])
		tensorIdx = np.vstack((tensorIdx, [[axis0Id, axis1Id]]))
		tensorVal = np.vstack((tensorVal, [[int(row[valueIdx])]]))
	tensorIdx = np.delete(tensorIdx, (0), axis=0)
	tensorVal = np.delete(tensorVal, (0), axis=0)
	f.close()
	tenX = sptensor.sptensor(tensorIdx, tensorVal, np.array([len(axis0Dict), len(axis1Dict)]))
	axisDict = {0: axis0Dict, 1: axis1Dict}
	return tenX, axisDict

## Save a single tensor into a file
def saveSingleTensor(X, axisDict, patClass, outfilePattern):
	"""  
	Save a single tensor (the original data, axis information, and classification)
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
	## save tensor via sptensor
	X.saveTensor(outfilePattern.format("data"))
	tensorInfo = shelve.open(outfilePattern.format("info"), "c")
	tensorInfo[AXIS] = axisDict
	tensorInfo[CLASS] = patClass
	tensorInfo.close()

## Load a single tensor and the axis information
def loadSingleTensor(inFilePattern):
	X = sptensor.loadTensor(inFilePattern.format("data"))
	tensorInfo = shelve.open(inFilePattern.format("info"), "r")
	axisDict = tensorInfo[AXIS]
	classDict = tensorInfo[CLASS]
	tensorInfo.close()
	return X, axisDict, classDict

def saveMultiTensor(X, sharedModes, axisDict, patClass, outfilePattern):
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
    outfilePattern : the pattern for the output format, note that {0} is necessary as 2 files are produced
    """
    outfile = file(outfilePattern.format("data"), "wb")
    np.save(outfile, len(X)) ## save the length so we know the number of tensors
    for tx in X:
        np.save(outfile, tx.subs)
        np.save(outfile, tx.vals)
        np.save(outfile, tx.shape)
    
    np.save(outfile, sharedModes) # then save the shared modes between the tensors
    outfile.close()
    
    ## store the information into a datafile
    tensorInfo = shelve.open(outfilePattern.format("info"), "c")
    tensorInfo[AXIS] = axisDict
    tensorInfo[CLASS] = patClass
    tensorInfo.close()

def loadMultiTensor(inFilePattern):
    """ 
    Load the list of tensors from this input file format
    
    Parameters
    ------------
    inFilePattern : the input file pattern for the 2 files with the tensor data and axis information
    
    Output
    -----------
    X : the list of tensors in the file
    sharedModes : the 2-d array with the shared modes location
    axisDict : the axis information for all the tensors
    patClass : the patient cohort information
    """
    infile = file(inFilePattern.format("data"), "rb")
    lenX = np.load(infile)
    X = []
    for i in range(lenX):
        subs = np.load(infile)
        vals = np.load(infile)
        siz = np.load(infile)
        X.append(sptensor.sptensor(subs, vals, siz))
    sharedModes = np.load(infile)
    tensorInfo = shelve.open(inFilePattern.format("info"), "r")
    axisDict = tensorInfo[AXIS]
    patClass = tensorInfo[CLASS]
    tensorInfo.close()
    
    return X, sharedModes, axisDict, patClass

## Read the file with the class information
def readClassFile(filename, patDict, patIdx, classIdx):
	patClass = OrderedDict()
	f = open(filename, "rb")
	for row in csv.reader(f):
		if not patDict.has_key(row[patIdx]):
			print "Doesn't have: " + row[patIdx]
			continue
		patId = patDict.get(row[patIdx])
		patClass[patId] = int(row[classIdx])
	f.close()
	return patClass



def getSingleMongoFormat(M, axisDict):
	""" 
	Flatten the CP decomposition so that it can be written into a mongo format
	
    """
	output = {}
	lmbda = M.lmbda
	idx = np.flatnonzero(lmbda)
	output['lambda'] = np.column_stack((idx, lmbda[idx])).tolist()
	for axis in axisDict.keys():
		tTuple = (axis)
		print tTuple
        factMatrix = M.U[axis]
        tmp = np.nonzero(factMatrix)
        output[tTuple] = np.column_stack((tmp[0], np.array(axisDict[axis], dtype="S100")[tmp[0]], tmp[1], factMatrix[tmp])).tolist()
	return output


def getMultiMongoFormat(M, modes, axisDict):
	output = {}
	# get the lambda's First
	lmbda = M[0].lmbda
	idx = np.flatnonzero(lmbda)
	output['lambda'] = np.column_stack((idx, lmbda[idx])).tolist()
	for m in modes:
		tenM = m.flatten().tolist()
		print tenM
        tTuple = (tenM[0], tenM[1])
        print tTuple
        factMatrix = M[tenM[0]].U[tenM[1]]
        tmp = np.nonzero(factMatrix)
        output[tTuple] = np.column_stack((tmp[0], np.array(axisDict[tTuple], dtype="S100")[tmp[0]], tmp[1], factMatrix[tmp])).tolist()
	return output