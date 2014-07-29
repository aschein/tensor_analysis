import numpy as np
import sptensor
import shelve
from collections import OrderedDict

def parseFile(filename, patIdx, medIdx, diagIdx, labelIdx, delim="|"):
    """ 
    Parse a csv file using the delimiter and the appropriate columns of interest.
    The resultant sparse tensor has patient on the 0th mode, diagnosis on the 1st mode,
    and medications on the 2nd mode.
    
    Tensor info contains the axis information for each mode.
    """
    print "Creating the tensor for " + filename

    patList = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
    medList = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
    diagList = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
    patClass = OrderedDict(sorted({}.items(), key=lambda t:t[1]))

    ## storing tensor class as empty array
    tensorIdx = np.array([[0, 0, 0]])
    datfile = open(filename)

    for i, line in enumerate(datfile):
        line = line.rstrip('\r\n')
        parse = line.split(delim)
        
        # insert them into the list if necessary
        if not patList.has_key(parse[patIdx]):
            patList[parse[patIdx]] = len(patList)
            patClass[parse[patIdx]] = parse[labelIdx]
        if not diagList.has_key(parse[diagIdx]):
            diagList[parse[diagIdx]] = len(diagList)
        if not medList.has_key(parse[medIdx]):
            medList[parse[medIdx]] = len(medList)
        
        patId = patList.get(parse[patIdx])
        diagId = diagList.get(parse[diagIdx])
        medId = medList.get(parse[medIdx])
    
        # we know the first one is already mapped
        if i > 1:
            tensorIdx = np.append(tensorIdx, [[patId, diagId, medId]], axis=0)

    tensorVal = np.ones((tensorIdx.shape[0], 1))
    # initialize size
    siz = np.array([len(patList), len(diagList), len(medList)])
    X = sptensor.sptensor(tensorIdx, tensorVal, siz)
    
    tensorInfo = {}
    tensorInfo['axis'] = [patList.keys(), diagList.keys(), medList.keys()]
    tensorInfo['pat'] = patList.keys()
    tensorInfo['med'] = medList.keys()
    tensorInfo['diag'] = diagList.keys()
    tensorInfo['class'] = patClass.values()
      
    return X, tensorInfo

def saveTensor(X, info, outFilename):
    """ 
    Save the tensor into 2 files ({0} in the filename is filled with data and info)
    One file contains the raw values, and the other the axis information
    """
    ## store the tensor objects into a file
    X.saveTensor(outFilename.format("data"))  
    # store the information into a data file
    tensorInfo = shelve.open(outFilename.format("info"), "c")
    tensorInfo['axis'] = info['axis']
    tensorInfo['pat'] = info['pat']
    tensorInfo['med'] = info['med']
    tensorInfo['diag'] = info['diag']
    tensorInfo['class'] = info['class']
    tensorInfo.close()