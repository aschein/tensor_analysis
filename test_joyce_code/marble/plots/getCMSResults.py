from pymongo import MongoClient
from collections import OrderedDict
import numpy as np
import json

mongoC = MongoClient()
db = mongoC.marble
cmsdb = db.cms

dataOut = []
### expt range
exptRange = range(0, 1000, 100)
for expt in exptRange:
	for sample in range(10):
		exptRun = cmsdb.find_one({"expt": expt+sample})
		if exptRun == None:
			continue
		order = exptRun['order']
		auc = exptRun['auc']
		d = dict(zip(order, auc))
		dataOut.append({"expt": expt+sample, "auc": d, "R": exptRun['R']})

with open('cms-results.json', 'w') as outfile:
	json.dump(dataOut, outfile)