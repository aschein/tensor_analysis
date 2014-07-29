'''
A tensor stored as a decomposed Kruskal operator.
The code is the python implementation of the @ktensor folder in the MATLAB Tensor Toolbox
'''
import numpy as np
import khatrirao
import itertools
import tensor

class ktensor:
    shape = None
    lmbda = None
    U = None
    R = 0
    
    def __init__(self, lmbda, U):
        """ 
        The tensor object stored as a Kruskal operator (decomposed).
        X = sum_r [lambda_r outer(a_r, b_r, c_r)].
        The columns of matrices A,B,C are the associated a_r, b_r, c_r
        """
        self.lmbda = lmbda
        self.R = len(lmbda)
        self.U = U
        self.shape = [len(self.U[r]) for r in range(len(U))]
        self.shape = tuple(self.shape)
        
    def __str__(self):
        ret = "Kruskal decomposition tensor with size {0}\n".format(self.shape)
        ret += "Lambda: {0}\n".format(self.lmbda)
        for i in range(len(self.U)):
            ret += "U[{0}] = {1}\n".format(i, self.U[i])
        return ret

    def arrange(self):
        """ 
        Normalizes the columns of the factor matrices and then sorts the
        components by magnitude, greatest to least.
        """
        self.normalize()
        self.sort_components()
            
    def fixsigns(self):
        """ 
        For each vector in each factor, the largest magnitude entries of K
        are positive provided that the sign on pairs of vectors in a rank-1
        component can be flipped.
        """
        for r in range(self.R):
            sgn=np.zeros(self.ndims())
            for n in range(self.ndims()):
                idx = np.argmax(np.abs(self.U[n][:, r]))
                sgn[n] = np.sign(self.U[n][idx, r])
            negidx = np.nonzero(sgn == -1)[0]
            nflip = 2 * np.floor(len(negidx) / 2)
            
            for i in np.arange(nflip):
                n = negidx[i]
                self.U[n][:,r] = -self.U[n][:,r]
        return

    def innerprod(self, Y):
        """
        Compute the inner product between this tensor and Y.
        If Y is a ktensor, the inner product is computed using inner products of the factor matrices.
        Otherwise, the inner product is computed using ttv with all of the columns of X's factor matrices
        """
        res = 0
        if (Y.__class__ == ktensor):
            M = np.outer(self.lmbda, Y.lmbda)
            for n in range(self.ndims()):
                M = np.multiply(M, np.inner(self.U[n], Y.U[n]))
            res = np.sum(M)
        else:
            vecs = [{} for i in range(self.ndims())]
            for r in range(self.R):
                for n in range(self.ndims()):
                    vecs[n] = self.U[n][:,r]
                res = res + self.lmbda[r]*Y.ttv(vecs, range(self.ndims()))
        return res
    
    def ndims(self):
        return len(self.U)
    
    def norm(self):
        """ returns the Frobenius norm of the tensor."""
        coefMatrix = np.outer(self.lmbda, self.lmbda)
        for i in range(self.ndims()):
            coefMatrix = np.multiply(coefMatrix, np.dot(self.U[i].transpose(),self.U[i]))
        return np.sqrt(np.abs(np.sum(coefMatrix)))
    
    def normalize(self, normtype=2):
        """" 
        Normalize the column of each factor matrix U, absorbing excess weight into lambda.
        Also ensures lambda is positive. 
        """
        ## Normalize the matrices
        for n in range(self.ndims()):
            colNorm = np.apply_along_axis(np.linalg.norm, 0, self.U[n], normtype)
            zeroNorm = np.where(colNorm == 0)[0]
            colNorm[zeroNorm] = 1
            ## multiply lambda by norm
            self.lmbda = self.lmbda * colNorm
            self.U[n] = self.U[n] / colNorm[np.newaxis,  :]
        idx = np.count_nonzero(self.lmbda < 0)
        if idx > 0:
            for i in np.nonzero(self.lmbda < 0):
                self.U[0][:, i] = -1 * self.U[0][:, i]
                self.lmbda[i] = -1*self.lmbda[i]
    
    def normalize_mode(self, mode, normtype):
        """Normalize the ith factor using the norm specified by normtype"""
        colNorm = np.apply_along_axis(np.linalg.norm, 0, self.U[mode], normtype)
        zeroNorm = np.where(colNorm == 0)[0]
        colNorm[zeroNorm] = 1
        self.lmbda = self.lmbda * colNorm
        self.U[mode] = self.U[mode] / colNorm[np.newaxis,  :]
    
    def normalize_sort(self, normtype=2):
        """"Normalize the column of each factor and
        sort each component/rank by magnitude greatest->smallest"""
        self.normalize(normtype)
        self.sort_components()
        
    def normalize_absorb(self, mode, normtype):
        """ 
        Normalize all the matrices using the norm specified by normtype and 
        then absorb all the lambda magnitudes into the factors.
        """
        self.normalize(normtype)
        self.U[mode] = np.inner(self.U[mode], np.diag(self.lmbda))
        self.lmbda = np.ones(self.R)
    
    def permute(self, order):
        """
        Rearranges the dimensions of the ktensor so the order is
        specified by the vector order.
        """
        return ktensor(self.lmbda, self.U[order])
        
    def redistribute(self, mode):
        """ 
        Distribute the lambda values to a specified mode.
        Lambda vector is set to all ones, and the mode n takes on the values
        """
        self.U[mode] = self.U[mode] * self.lmbda[np.newaxis, :]
        self.lmbda = np.ones(self.R)
            
    ######################### SCORING FUNCTIONS #################################
    def __calculateCongruences(self, B):
        # first make sure both are normalized
        self.normalize(2)
        B.normalize(2)
        # compute the product for each pair of matrices for each mode
        rawC = [abs(np.dot(self.U[n].transpose(),B.U[n])) for n in range(self.ndims())]
        # compute the penalty based on difference in lambdas
        rawP = np.zeros((self.R, self.R))
        for ra in range(self.R):
            for rb in range(B.R):
                rawP[ra,rb] = 1 - (abs(self.lmbda[ra] - self.lmbda[rb]) / max(self.lmbda[ra], B.lmbda[rb]))
        C = np.ones((self.R, self.R))
        for n in range(self.ndims()):
            C = C * rawC[n]
        C = rawP * C
        return rawC, rawP, C
    
    def greedy_fms(self, B):
        """
        Compute the factor match score based on greedy search of the permutations
        So the best matching factor first, then next, etc.
        """
        rawC,rawP, C = self.__calculateCongruences(B)
        selfR = []
        BR =[]
        for r in range(B.R):
            maxIdx = np.unravel_index(C.argmax(), C.shape)
            selfR.append(maxIdx[0])
            BR.append(maxIdx[1])
            C[maxIdx[0],:] = 0
            C[:, maxIdx[1]] = 0
        sc = {'OrigOrder': selfR, 'OtherOrder': BR, 'Lambda': rawP[selfR, BR].tolist()}
        selfR = np.array(selfR)
        BR = np.array(BR)
        for n in range(self.ndims()):
            sc[str(n)] = (rawC[n][selfR, BR]).tolist()
        return sc
    
    def top_fms(self, B, topF=10):
        """
        Compute the factor match score based on the best possible permutation 
        of the topF factors.
        """
        # create a new ktensor with just the top ones
        idx = range(topF)
        selfTOP = ktensor(self.lmbda[idx], [self.U[n][:, idx] for n in range(self.ndims())])
        BTOP = ktensor(B.lmbda[idx], [B.U[n][:, idx] for n in range(B.ndims())])
        return selfTOP.fms(BTOP)
        
    def fms(self, B):
        """ 
        Compute the factor match score based on the best possible permutation
        """
        rawC,rawP, C = self.__calculateCongruences(B)
        # generate all possible permutations of the factors for B
        combos = list(itertools.permutations(range(self.R)))
        # calculate the scores for each one
        scores = np.zeros((len(combos), B.R))
        for i in range(len(combos)):
            for rb in range(B.R):
                scores[i,rb] = C[combos[i][rb], rb]
        # sum across the rows
        bscore = np.sum(scores, axis=1) / B.R
        # find the combination that gives the best score
        cb = np.argmax(bscore)
        #print(bscore[cb])
        sc = np.column_stack((np.repeat(-1, self.R), range(self.R), combos[cb], rawP[combos[cb], range(self.R)]))
        for n in range(self.ndims()):
            tmp = np.column_stack((np.repeat(n, self.R), range(self.R), combos[cb], rawC[n][combos[cb], range(self.R)]))
            sc = np.vstack((sc, tmp))
        # return the raw highest score
        return sc
    
    @staticmethod
    def computeJaccardIndex(f1, f2):
        """ Compute the Jaccard index /similarity between 2 factor matrices """
        num = np.sum(np.logical_and(f1, f2), axis=0).astype(float)
        denom = np.add(num, np.sum(np.logical_xor(f1, f2), axis=0))
        # fix the 0's
        denom = np.maximum(0.001*np.ones(len(denom)), denom)
        return num / denom
    
    @staticmethod
    def __calculateFOSCongruences(A, B, R):
        N = len(A)
        rawC = [np.zeros((R, R)) for n in range(N)]
        for r in range(R):
            for n in range(N):
                tmpA = np.tile(A[n][:, r], (R,1)).transpose()
                rawC[n][r, :] = ktensor.computeJaccardIndex(tmpA, B[n])
        C = np.zeros((R, R))
        for n in range(N):
            C = C + rawC[n]
        return rawC, C
    
    @staticmethod
    def __binarizeFactors(A, B):
        A.normalize_sort(1)
        B.normalize_sort(1)
        binA = [(A.U[n] != 0) for n in range(len(A.U))]
        binB = [(B.U[n] != 0) for n in range(len(B.U))]
        return binA, binB
        
    def greedy_fos(self, B):
        """
        Compute the factor over score based on greedy search of the permutations
        So start with the first factor and work your way downards
        """
        binSelf, binB = ktensor.__binarizeFactors(self, B)
        rawC, C = ktensor.__calculateFOSCongruences(binSelf, binB, self.R)
        selfR = []
        BR =[]
        for r in range(self.R):
            maxIdx = np.unravel_index(C.argmax(), C.shape)
            selfR.append(maxIdx[0])
            BR.append(maxIdx[1])
            C[maxIdx[0],:] = 0
            C[:, maxIdx[1]] = 0
        sc = {'OrigOrder': selfR, 'OtherOrder': BR}
        selfR = np.array(selfR)
        BR = np.array(BR)
        for n in range(self.ndims()):
            sc[str(n)] = (rawC[n][selfR, BR]).tolist()
        return sc
    
    def top_fos(self, B, topF=10):
        """
        Compute the factor overlap score based on the best possible permutation 
        of the topF factors.
        """
        self.normalize_sort(1)
        B.normalize_sort(1)
        # create a new ktensor with just the top ones
        idx = range(topF)
        selfTOP = ktensor(self.lmbda[idx], [self.U[n][:, idx] for n in range(self.ndims())])
        BTOP = ktensor(B.lmbda[idx], [B.U[n][:, idx] for n in range(B.ndims())])
        return selfTOP.fos(BTOP)

    def fos(self, B):
        """ Factor overlap score """
        # calculate all possible congruences
        binSelf, binB = ktensor.__binarizeFactors(self, B)
        rawC, C = ktensor.__calculateFOSCongruences(binSelf, binB, self.R)
        combos = list(itertools.permutations(range(self.R)))
        # calculate the scores for each one
        scores = np.zeros((len(combos), B.R))
        for i in range(len(combos)):
            for rb in range(B.R):
                scores[i,rb] = C[combos[i][rb], rb]
        # sum across the rows
        bscore = np.sum(scores, axis=1) / B.R
        # find the combination that gives the best score
        cb = np.argmax(bscore)
        sc = np.zeros((1, 4))
        for n in range(self.ndims()):
            tmp = np.column_stack((np.repeat(n, self.R), range(self.R), combos[cb], rawC[n][combos[cb], range(self.R)]))
            sc = np.vstack((sc, tmp))
        sc = np.delete(sc, (0), axis=0)
        return sc
        
    def sort_components(self):
        """ Sort the ktensor components by magnitude, greatest to least."""
        sortidx = np.argsort(self.lmbda)[::-1];
        self.lmbda = self.lmbda[sortidx];
        # resort the u's
        for i in range(self.ndims()):
            self.U[i] = self.U[i][:, sortidx];

    def toTensor(self):
        """Convert this to a dense tensor"""
        tmp = khatrirao.khatrirao_array(self.U, True)
        data = np.inner(self.lmbda, tmp);
        return tensor.tensor(data, self.shape);
    
    def ttv(self, v, dims):
        """ 
        Computes the product of the Kruskal tensor with the column vector along
        specified dimensions.
        
        Parameters
        ----------
        v - column vector 
        dims - dimensions to multiply the product

        Returns
        -------
        out : 
        """
        (dims,vidx) = tools.tt_dimscheck(dims, self.ndims(), len(v));
        remdims = np.setdiff1d(range(self.ndims()), dims);
        
        ## Collapse dimensions that are being multiplied out
        newlmbda = self.lmbda;
        for i in range(self.ndims()):
            newlmbda = np.inner(np.inner(self.U[dims[i]], v[vidx[i]]));
        
        if len(remdims) == 0:
            return np.sum(newlmbda);
        
        return ktensor(newlmbda, self.u[remdims]);
    
    def writeRawFile(self, filename):
        ## store all the stuff in raw format
        outfile = file(filename, "wb")
        np.save(outfile, self.lmbda)
        np.save(outfile, self.ndims())
        for n in range(self.ndims()):
            np.save(outfile, self.U[n])
        outfile.close()
        
    ### Mathematic and Logic functions
    def __add__(self, other):
        if (other.__class__ != ktensor):
            raise ValueError("Must be two ktensors for addition");
        if (not self.shape == other.shape):
            raise ValueError("Two ktensors must have the same shape");
        lmbda = self.lmbda + other.lmbda;
        U = []
        for m in range(self.ndims()):
            U.append(np.concatenate((self.U[m], other.U[m]), axis=1));
        return ktensor(lmbda, U);

    def __sub__(self, other):
        if (other.__class__ != ktensor):
            raise ValueError("Must be two ktensors for subtraction");
        if (not self.shape == other.shape):
            raise ValueError("Two ktensors must have the same shape");
        lmbda = self.lmbda + [-n for n in other.lmbda];
        U = []
        for m in range(self.ndims()):
            U.append(np.concatenate((self.U[m], other.U[m]), axis=1));
        return ktensor(lmbda, U);
        
    def __eq__(self, other):
        if (other == None):
            return False
        if(other.__class__ == ktensor):
            raise ValueError("Must be two ktensors for comparison");
        if(self.shape != other.shape):
            raise ValueError("Size Mismatch");
        tf = self.lmbda == other.lmbda;
        if tf:
            # if lambda is true, then continue onto the other components
            for m in range(self.ndims()):
                tf = tf and np.min(np.equal(self.U[m], other.U[m]));
        return tf;
    
def loadTensor(filename):
    """ Load the tensor from a file """
    infile = file(filename, "rb")
    lmbda = np.load(infile)
    N = np.load(infile)
    U = []
    for n in range(N):
        U.append(np.load(infile))
    return ktensor(lmbda, U)

def copyTensor(X):
    """Create a deep copy of the tensor"""
    return ktensor(X.lmbda.copy(), [X.U[n].copy() for n in range(X.ndims())])
