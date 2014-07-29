import ktensor
import numpy as np

R = 4
A = ktensor.ktensor(np.ones(R), [np.random.rand(5,R), np.random.rand(5,R), np.random.rand(2,R)])
B = ktensor.ktensor(np.ones(R), [np.random.rand(5,R), np.random.rand(5,R), np.random.rand(2,R)])

rawFMS = A.fms(B)
topFMS = A.top_fms(B, 2)
greedFMS = A.greedy_fms(B)

print rawFMS, topFMS, greedFMS

np.random.seed(10)
A = ktensor.ktensor(np.ones(R), [np.random.randn(5,R), np.random.randn(5,R), np.random.randn(2,R)])
A.U = [np.multiply((A.U[n] > 0).astype(int), A.U[n])  for n in range(A.ndims())]
B = ktensor.ktensor(np.ones(R), [np.random.randn(5,R), np.random.randn(5,R), np.random.randn(2,R)])
B.U = [np.multiply((B.U[n] > 0).astype(int), B.U[n]) for n in range(B.ndims())]

rawFOS = A.fos(B)
topFOS = A.top_fos(B, 2)
greedFOS = A.greedy_fos(B)

print rawFOS, topFOS, greedFOS