'''
Base code from pytensor: the python implementation of MATLAB based tensor code
https://code.google.com/p/pytensor

A sparse representation of the tensor object, storing only nonzero entries.
The code is the python implementation of the @sptensor folder in the MATLAB Tensor Toolbox
'''
import numpy;
import sptenmat;
import tenmat;
import tensor;
from scipy import sparse;
import tools;

class sptensor:
    subs = None;
    vals = None;
    shape = None;
    func = sum.__call__;
    
    def __init__(self,
                 subs,
                 vals,
                 shape = None,
                 func = sum.__call__):
        """
        Create a sptensor object. The subs array specifies the nonzero entries
        in the tensor, with the kth row of subs corresponding to the kth entry 
        in vals.
        
        Parameters
        ----------
        subs - p x n array specifying the subscripts of nonzero entries
        vals - the corresponding value of the nonzero entries
        shape - the shape of the tensor object.
        func - accumulation function for repeated subscripts

        Returns
        -------
        out : sparse tensor object
        
        """
        
        if(subs.__class__ == list):
            subs = numpy.array(subs);
        if(vals.__class__ == list):
            vals = numpy.array(vals);
        if(shape.__class__ == list):
            shape = numpy.array(shape);
            
        
        if not(tools.tt_subscheck(subs)):
            raise ValueError("Error in subscripts");
        if not(tools.tt_valscheck(vals)):
            raise ValueError("Error in values");
        if (shape != None and not tools.tt_sizecheck(shape)):
            raise ValueError("Error in shape");
        
        if(vals.size != 0 and vals.size != 1 and len(vals) != len(subs)):
            raise ValueError("Number of subscripts and values must be equal");
        
        if (shape == None):
            self.shape = tuple(subs.max(0) + 1);
        else:
            self.shape = tuple(shape);
        
        # if func is given by user
        if(func != None):
            self.func = func;
        
        if(subs.size == 0):
            nzsub = numpy.array([]);
            nzval = numpy.array([]);
        else:
            (newsub, loc) = uniquerows(subs);
            newval = numpy.ndarray([len(newsub), 1]);
            newval.fill(0);
            
            for i in range(0, len(loc)):
                newval[(int)(loc[i])] = func(vals[i], newval[(int)(loc[i])]);
        
            nzsub = newsub.tolist();
            nzval = newval.tolist();
            
            
            i = 0;
            while (i < len(nzsub)):
                if(nzval[i][0] == 0):
                    nzsub.remove(nzsub[i]);
                    nzval.remove(nzval[i]);
                else:
                    i = i+1;
                
        self.subs = numpy.array(nzsub);
        self.vals = numpy.array(nzval);    
        
    def ndims(self):
        """ returns the number of dimension of the tensor"""
        return len(self.shape);
        
    def dimsize(self, ind):
        """ returns the size of the specified dimension.
        Same as shape[ind]."""
        return self.shape[ind];
    
    def mttkrp(self, U, n):
        """ Matricized tensor times Khatri-Rao product for sparse tensor.
        
        Calculates the matrix product of the n-mode matricization of X with
        the Khatri-Rao product of all entries in U except the nth.
        A series of TTV operations are performed rather than forming the Khatri-Rao product.
        
        Parameters
        ----------
        U - factorization
        n - the mode not to calculate

        Returns
        -------
        out : Khatri-Rao product as a numpy array
        """
        N  = self.ndims();
        if (n == 0):
            R = U[1].shape[1];
        else:
            R = U[0].shape[1];
        V = numpy.zeros((self.shape[n], R));
        for r in range(R):
            Z = [{} for i in range(N)];
            dim = numpy.concatenate((numpy.arange(0, n), numpy.arange(n+1, N)));
            for i in dim:
                Z[i] = U[i][:,r];
            V[:,r] = self.ttv(Z, dim).tondarray();
        return V;
    
    def norm(self):
        """ returns the Frobenius norm of the tensor."""
        return numpy.linalg.norm(self.vals);
    
    def nnz(self):
        """returns the number of non-zero elements in the sptensor"""
        return len(self.subs);
    
    def totensor(self):
        """returns a new tensor object that contains the same values"""
        temp = numpy.ndarray(self.shape);
        temp.fill(0);
        
        for i in range(0, len(self.vals)):
            temp.put(tools.sub2ind(self.shape, self.subs[i])[0], self.vals[i][0]);
        
        return tensor.tensor(temp, self.shape);
    
    def __str__(self):
        if (self.nnz() == 0):
            return "all zero sparse tensor of size {0}".format(self.shape);
        else:
            ret = "sparse tensor of size {0} with {1} non-zero elements\n".format(self.shape, self.nnz());
            for i in range (0, len(self.subs)):
                ret += "\n{0} {1}".format(self.subs[i], self.vals[i]);
            return ret;

    def copy(self):
        return sptensor(self.subs.copy(), self.vals.copy(),
                        self.shape, self.func);


    
      
    def permute(self, order):
        """returns a new sptensor permuted by the given order"""
        if (order.__class__ == list):
            order = numpy.array(order);
            
        if(self.ndims() != len(order)):
            raise ValueError("invalid permutation order")
        
        sortedorder = order.copy();
        sortedorder.sort();
        
        if not ((sortedorder == numpy.arange(len(self.shape))).all()):
            raise ValueError("invalid permutation order");
        
        neworder = numpy.arange(len(order)).tolist();
        newsiz = list(self.shape);
        newval = self.vals.copy();
        newsub = self.subs.copy();

        for i in range(0,len(order)-1):
            index = tools.find(neworder, order[i]);            
            
            for s in newsub:
                temp = s[i];
                s[i] = s[index];
                s[index] = temp;
            
            temp = newsiz[i];
            newsiz[i] = newsiz[index];
            newsiz[index] = temp;
            
            temp = neworder[i];
            neworder[i] = neworder[index];
            neworder[index] = temp;
            
        return sptensor(newsub, newval, newsiz, self.func);
    
    
    
    def ttm(self, mat, dims = None, option = None):
        """ computes the sptensor times the given matrix.
        arrs is a single 2-D matrix/array or a list of those matrices/arrays."""
        
        if(dims == None):
            dims = range(0,self.ndims());
        
        #Handle when arrs is a list of arrays
        if(mat.__class__ == list):
            if(len(mat) == 0):
                raise ValueError("the given list of arrays is empty!");
            
            (dims,vidx) = tools.tt_dimscehck(dims, self.ndims(), len(mat));
            
            Y = self.ttm(mat[vidx[0]],dims[0],option);
            for i in range(1, len(dims)):
                Y = Y.ttm(mat[vidx[i]],dims[i],option);
                
            return Y;                
        
        if(mat.ndim != 2):
            raise ValueError ("matrix in 2nd armuent must be a matrix!");

        if(option != None):
            if (option == 't'):
                mat = mat.transpose();
            else:
                raise ValueError ("unknown option.");          
        
        
        if(dims.__class__ == list):
            if(len(dims) != 1):
                raise ValueError("Error in number of elements in dims");
            else:
                dims = dims[0];
        
        if(dims < 0 or dims > self.ndims()):
            raise ValueError ("Dimension N must be between 1 and num of dimensions");
        
        #Check that sizes match
        if(self.shape[dims] != mat.shape[1]):
            raise ValueError ("size mismatch on V");
        
        #Compute the new size
        newsiz = list(self.shape);
        newsiz[dims] = mat.shape[0];
        
        #Compute Xn
        Xnt = sptenmat.sptenmat(self,None,[dims],None,'t');
        rdims = Xnt.rdims;
        cdims = Xnt.cdims;
        
        I = [];
        J = [];
        for i in range(0, len(Xnt.subs)):
            I.extend([Xnt.subs[i][0]]);
            J.extend([Xnt.subs[i][1]]);
        
        
        Z = (sparse.coo_matrix((Xnt.vals.flatten(),(I,J)),
            shape = (tools.getelts(Xnt.tsize, Xnt.rdims).prod(),
                     tools.getelts(Xnt.tsize, Xnt.cdims).prod()))
             * mat.transpose());
        
        Z = tensor.tensor(Z,newsiz).tosptensor();
        
        
        if(Z.nnz() <= 0.5 * numpy.array(newsiz).prod()):
            Ynt = sptenmat.sptenmat(Z, rdims, cdims);
            return Ynt.tosptensor();
        else:
            Ynt = tenmat.tenmat(Z.totensor(), rdims, cdims);
            return Ynt.totensor();
    
    def ttv(self, v, dims):
        """ 
        Computes the product of this tensor with the column vector along
        specified dimensions.
        
        Parameters
        ----------
        v - column vector 
        d - dimensions to multiply the product

        Returns
        -------
        out : a sparse tensor if 50% or fewer nonzeros
        """
        
        (dims, vidx) = tools.tt_dimscheck(dims, self.ndims(), len(v));
        remdims = numpy.setdiff1d(range(self.ndims()), dims);
        newvals = self.vals;
        subs = self.subs;
        
        # Multiple each value by the appropriate elements of the appropriate vector
        for n in range(len(dims)):
            idx = subs[:, dims[n]]; # extract indices for dimension n
            w = v[vidx[n]];         # extract nth vector
            bigw = w[idx];          # stretch out the vector
            newvals = numpy.multiply(newvals.flatten(), bigw);
            
        # Case 0: all dimensions specified - return the sum
        if len(remdims) == 0: 
            c = numpy.sum(newvals);
            return c;
        
        # Otherwise figure out the subscripts and accumulate the results
        newsubs = self.subs[:, remdims];
        newsiz = numpy.array(self.shape)[remdims];
        
        # Case 1: return a vector
        if len(remdims) == 1:
            c = tools.accum_np(newsubs, newvals, newsiz[0]);
            #if numpy.count_nonzero(c) < 0.5*newsiz[0]:
            #    c = sptensor.sptensor(numpy.arange(newsiz[0]).reshape(newsiz[0],1), c.reshape(newsiz[0],1));
            #else:
            c = tensor.tensor(c, newsiz);
            return c;
        
        # Case 2: result is a multi-way array
        c = sptensor.sptensor(newsubs, newvals.reshape(len(newvals), 1), newsiz);
        # check to see if it's dense
        if c.nnz() > 0.5*numpy.prod(c.shape):
            return c.totensor();
        return c;
    
    def tondarray(self):
        """returns an ndarray that contains the data of the sptensor"""
        return self.totensor().tondarray();
    
    def saveTensor(self, filename):
        outfile = file(filename, "wb")
        numpy.save(outfile, self.subs)
        numpy.save(outfile, self.vals)
        numpy.save(outfile, self.shape)
        outfile.close()
    
    def __add__(self, other):
        if (other.__class__ == sptensor):
            if (not self.shape == other.shape):
                raise ValueError("Two sparse tensors must have the same shape");
            return sptensor(self.subs.tolist() + other.subs.tolist(),
                        self.vals.tolist() + other.vals.tolist(), self.shape);
        
        #other is a tensor or a scalar value
        return self.totensor() + other;

    def __sub__(self, other):
        if (other.__class__ == sptensor):
            if (not self.shape == other.shape):
                raise ValueError("Two sparse tensors must have the same shape");
            return sptensor(self.subs.tolist() + other.subs.tolist(),
                        self.vals.tolist() + (-other.vals).tolist(), self.shape);
            
        #other is a tensor or a scalar value
        return self.totensor() - other;
        

    def __eq__(self, oth):
        if(oth.__class__ == sptensor):
            if(self.shape != oth.shape):
                raise ValueError("Size Mismatch");
            sub1 = self.subs;
            sub2 = oth.subs;
            usub = union(sub1, sub2);
            ret = (tools.allIndices(oth.shape));
            
        elif(oth.__class__ == tensor):
            return self.__eq__(oth.tosptensor());
            
        elif(oth.__class__ == int or oth.__class__ == float or oth.__class__ == bool):
            newvals = (self.vals == oth);
            newvals = booltoint(newvals);
            return sptensor(self.subs, newvals, self.size);
            
        else:
            raise ValueError("error");

    def __ne__(self, oth):
        pass
        

    def __mul__(self, scalar):
        """multiples each element by the given scalar value"""
        if(scalar.__class__ == numpy.ndarray or
           scalar.__class__ == tensor.tensor or
           scalar.__class__ == sptensor):
            raise ValueError("multiplication is only with scalar value. use ttm, ttv, or ttt instead.");
        return sptensor(self.subs.copy(), self.vals.copy()*scalar, self.shape);
    
    def __pos__(self):
        pass; #do nothing
    def __neg__(self):
        return sptensor(self.subs.copy(), self.vals.copy() * -1 , self.shape);

def copyTensor(X):
    """Create a deep copy of the tensor"""
    return sptensor(X.subs, X.vals, X.shape)

def loadTensor(filename):
    """ Load a tensor from a file that has been saved using sptensor.saveTensor"""
    infile = file(filename, "rb")
    subs = numpy.load(infile)
    vals = numpy.load(infile)
    siz = numpy.load(infile)
    infile.close()
    return sptensor(subs, vals, siz)

def sptendiag(vals, shape = None):
    """special constructor, construct a sptensor with the given values in the diagonal"""
    #if shape is None or
    #number of dimensions of shape is less than the number of values given
    if (shape == None or len(shape) < len(vals)):
        shape = [len(vals)]*len(vals);
    else:
        shape = list(shape);
        for i in range(0, len(vals)):
            if(shape[i] < len(vals)):
                shape[i] = len(vals);
    
    subs = [];
    for i in range(0, len(vals)):
        subs.extend([[i]*len(shape)]);
    
    vals = numpy.array(vals).reshape([len(vals),1]);
    
    return sptensor(subs, vals, shape);

def uniquerows(arr):
    """ Given a 2D array, find the unique row and return the rows as 2-d array. """
    arr_dtype = arr.dtype.descr * arr.shape[1]
    struct = arr.view(arr_dtype)
    arr_uniq,idx = numpy.unique(struct, return_inverse=True)
    return (arr_uniq, idx);
    

#arr1, arr2: sorted list or sorted numpy.ndarray of subscripts.
#union returns the sorted union of arr1 and arr2.
def union(arr1, arr2):
    if(arr1.__class__ != list):
        a1 = arr1.tolist();
    else:
        a1 = arr1;
    if(arr2.__class__ != list):
        a2 = arr2.tolist();
    else:
        a2 = arr1;
    
    i = 0;
    j = 0;
    ret = numpy.array([]);
    
    if(len(a1) > 0):
        ret = [a1[i]];
        i = i+1;
    elif(len(a2) > 0):
        ret = [a2[j]];
        j = j+1;
    else:
        return numpy.array([[]]);
    
    while(i < len(a1) or j < len(a2)):
        if(i == len(a1)):
            ret = numpy.concatenate((ret, [a2[j]]), axis=0);
            j = j+1;
        elif(j == len(a2)):
            ret = numpy.concatenate((ret, [a1[i]]), axis=0);
            i = i+1;
        elif(a1[i] < a2[j]):
            ret = numpy.concatenate((ret, [a1[i]]), axis=0);
            i = i+1;
        elif(a1[i] > a2[j]):
            ret = numpy.concatenate((ret, [a2[j]]), axis=0);
            j = j+1;
        else:
            i = i+1;
    
    return ret;
    
