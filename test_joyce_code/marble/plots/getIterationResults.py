from pymongo import MongoClient
from collections import OrderedDict
import numpy as np
import json

mongoC = MongoClient()
db = mongoC.marble
simDB = db.simIter

dataOut = []
### expt range
exptRange = range(2010, 2020)
for expt in exptRange:
	exptRun = simDB.find_one({"exptID": expt})
	for inIter in [1, 2, 5, 10]:
		results = exptRun[str(inIter)]
		iterInfo = OrderedDict(sorted(results['iterInfo'].items(), key=lambda t: t[0]))
		totalUpdates = 0
		for k,v in iterInfo.items():
			## sum up the number of multiplicative updates
			totalUpdates = totalUpdates + np.sum(v['Iterations'])
		lastIter = iterInfo.keys()[-1]
		ll = iterInfo[lastIter]['LL']
		## get the total number of iterations
		totalIters = map(int, lastIter[1:-1].split(','))[0] + 1
		dataOut.append({"expt": expt, "iters": inIter, "comp": results['compTime'], 
			"totalOuter": totalIters, "totalMult": totalUpdates, "ll": ll,
			"fms": results['fms']})

with open('iterresults.json', 'w') as outfile:
	json.dump(dataOut, outfile)