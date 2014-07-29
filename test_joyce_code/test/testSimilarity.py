import ktensor
import numpy;
import decompTools

caseX = ktensor.loadTensor("results/apr-raw-1.dat")
controlX = ktensor.loadTensor("results/apr-raw-2.dat")
allX = ktensor.loadTensor("results/apr-raw-200.dat")

## since they don't share the same axis, remove one of the factors
caseX.U = [caseX.U[1], caseX.U[2]]
caseX = decompTools.zeroSmallFactors(caseX, 1e-2)
controlX.U = [controlX.U[1], controlX.U[2]]
controlX = decompTools.zeroSmallFactors(controlX, 1e-2)
allX.U = [allX.U[1], allX.U[2]]
allX = decompTools.zeroSmallFactors(allX, 1e-2)

fms = caseX.greedy_fms(controlX)
numpy.savetxt("plots/case-control.csv", fms, delimiter=",")
allFMS = caseX.greedy_fms(allX)
numpy.savetxt("plots/case-all.csv", allFMS, delimiter=",")
