import tensor
import sptensor
import numpy as np
import sim_APR
import ktensor

""" Test factorization of sparse matrix """
subs = np.array([[0,3,1], [1,0,1], [1,2,1], [1,3,1], [3,0,0]]);
vals = np.array([[1],[1],[1],[1],[3]]);
siz = np.array([5,5,2]) # 5x5x2 tensor
# do the tensor with the same one
X = [sptensor.sptensor(subs, vals, siz), sptensor.sptensor(subs, vals, siz)]
sharedModes = [np.array([[0,0], [1,1]])]
sapr = sim_APR.SAPR(X, 4, sharedModes)
sapr.factorize()

print sapr.M