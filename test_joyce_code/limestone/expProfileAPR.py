import argparse
import cProfile
import pstats

import CP_APR
import sptensor

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="input file to parse")
parser.add_argument("outputFile", help="output file for profile information")

parser.add_argument("-r", "--rank", type=int, help="rank of factorization", default=100)
parser.add_argument("-i", "--iters", type=int, help="Number of outer interations", default=100)
args = parser.parse_args()

########## Profile tensor factorization ###############
X = sptensor.loadTensor(args.inputFile)
## Profile
outputFile = args.outputFile
cProfile.run("CP_APR.cp_apr(X,R={0},tol=1e-2, maxiters={1}, maxinner=10)".format(args.rank, args.iters), filename=outputFile)

p = pstats.Stats(outputFile)
p.sort_stats('time').print_stats()