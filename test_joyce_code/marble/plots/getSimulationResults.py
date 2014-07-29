from pymongo import MongoClient
from collections import defaultdict
import json

mongoC = MongoClient()
db = mongoC.marble
simCol = db.sim

expt = 402
dataOut = []

exptRun = simCol.find_one({"exptID": expt})
for sample in range(10):
	sampleRun = exptRun[str(sample)]
	for k,v in enumerate(sampleRun['Order']):
		dataOut.append({"expt": expt, "sample": sample, "type": v, "comp": sampleRun['CompTime'][k], 
			"nonzero": sampleRun['NNZ'][k], "fms": sampleRun['FMS'][k], "FOS": sampleRun['FOS'][k]})
with open('simresults.json', 'w') as outfile:
	json.dump(dataOut, outfile)