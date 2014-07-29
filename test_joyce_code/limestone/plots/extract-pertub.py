from pymongo import MongoClient
from collections import OrderedDict
import numpy as np
import csv
import json

mongoC = MongoClient()
db = mongoC.limestone
perturbDB = db.perturb

csvOut = open('perturb-expt2.csv', 'wb')
writer = csv.writer(csvOut)

for expt in perturbDB.find({"rank":50, "type":"add+subtract"}):
	## create the numpy array
	for mode in range(3):
		tmp = np.column_stack((np.repeat(expt['expt'], 50), np.repeat(expt['noise'], 50), np.repeat(mode, 50)))
		tmp = np.column_stack((tmp, expt[str(mode)]))
		writer.writerows(tmp.tolist())

csvOut.close()

csvOut = open('perturb-expt1.csv', 'wb')
writer = csv.writer(csvOut)

for expt in perturbDB.find({"rank":50, "type":"add"}):
	## create the numpy array
	for mode in range(3):
		tmp = np.column_stack((np.repeat(expt['expt'], 50), np.repeat(expt['noise'], 50), np.repeat(mode, 50)))
		tmp = np.column_stack((tmp, expt[str(mode)]))
		writer.writerows(tmp.tolist())

csvOut.close()