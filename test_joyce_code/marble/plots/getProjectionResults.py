from pymongo import MongoClient
from collections import OrderedDict
import numpy as np
import json

mongoC = MongoClient()
db = mongoC.marble
projdb = db.simProj

dataOut = []
### expt range
exptRange = range(2000, 2010)
for expt in exptRange:
	exptRun = projdb.find_one({"exptID": expt})
	for projType in ["none", "full", "gamma1", "gamma2", "gamma"]:
		results = exptRun[projType]
		iterInfo = OrderedDict(sorted(results['iterInfo'].items(), key=lambda t: t[0]))
		totalUpdates = 0
		for k,v in iterInfo.items():
			## sum up the number of multiplicative updates
			totalUpdates = totalUpdates + np.sum(v['Iterations'])
		lastIter = iterInfo.keys()[-1]
		ll = iterInfo[lastIter]['LL']
		g = [0, 0, 0]
		if results.has_key('gamma'):
			g = results['gamma']
		## get the total number of iterations
		totalIters = map(int, lastIter[1:-1].split(','))[0] + 1
		dataOut.append({"expt": expt, "proj": projType, 
			"Mode 0": g[0], "Mode 1":g[1], "Mode 2":g[2],
			"nonzero": results['nnz'], "fms": results['fms']})

with open('projresults.json', 'w') as outfile:
	json.dump(dataOut, outfile)