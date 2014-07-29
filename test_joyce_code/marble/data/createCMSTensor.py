from collections import OrderedDict
import numpy as np
import csv
import json
import itertools
import re
from sklearn.utils import extmath

import sys
sys.path.append("../..")
import sptensor
import tensorTools

def loadJSON(fn):
	with open(fn, 'rb') as outfile:
		jsonDict = json.load(outfile)
		outfile.close()
	return jsonDict

def formatICD(code):
	"""
	Given an ICD-9 code that has no period, format for the period
	"""
	if code.isdigit():
		codeLen = len(code)
		if codeLen == 3:
			return code + ".00"
		elif codeLen == 4:
			return code[:3]+"."+ code[3:]+"0"
		elif codeLen == 5:
			return code[:3] + "." + code[3:]
	elif code[0] == 'V':
		return code[:3]+"."+code[3:]
	elif code[0] == 'E':
		return code[:4] + "."+code[4:]
	return code

def getDiagnosis(row, diagIdx, diagHier):
	"""
	Get all the diagnosis values from a row and weed out the empty ones
	"""
	diagArray = np.array(row)[diagIdx]
	diagArray = filter(None, diagArray)
	diagArray = [formatICD(k) for k in diagArray]
	diagCatArray = [formatLevel2(diagHier, k) for k in diagArray]
	diagCatArray = filter(None, diagCatArray)
	return diagArray, diagCatArray

def formatLevel2(codeHier, code):
	"""
	Given a code, get the level2 value
	"""
	if code in codeHier:
		return codeHier[code]['level2']
	return None

def getProc(row, procIdx, procHier):
	"""
	Get all the procedure codes from a row
	"""
	procArray = np.array(row)[procIdx]
	procArray = filter(None, procArray)
	procCatArray = [formatLevel2(procHier, k) for k in procArray]
	procCatArray = filter(None, procCatArray)
	return procArray, procCatArray

def parseCarrier(f):
	headerRow = True
	patientId = None
	claimId = None
	procHier = loadJSON("cpt.json")
	icdHier = loadJSON("icd.json")
	patDict = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
	diagDict = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
	procDict = OrderedDict(sorted({}.items(), key=lambda t:t[1]))
	## store the tensor index in an array
	tensorIdx = np.array([[0, 0, 0]])
	tensorVal = np.array([[0]])
	pid = 0

	for row in csv.reader(open(f, "rb")):
		if pid > 10000:
			break
		# For the header, we will get the values we need
		if headerRow:
			pidIdx = [i for i, item in enumerate(row) if re.search('DESYNPUF_ID', item)][0]
			claimIdx = [i for i, item in enumerate(row) if re.search('CLM_ID', item)][0]
			diagIdx = [ i for i, item in enumerate(row) if re.search('ICD9_DGNS', item)]
			hcpcsIdx = [ i for i, item in enumerate(row) if re.search('HCPCS_CD', item)]
			headerRow = False
			continue
		## get the diagnosis and procedure codes
		diagArray, diagCat = getDiagnosis(row, diagIdx, icdHier)
		for dc in set(diagCat):
			if not diagDict.has_key(dc):
				diagDict[dc] = len(diagDict)
		hcpcsArray, hcpcsCat = getProc(row, hcpcsIdx, procHier)
		for pc in set(hcpcsCat):
			if not procDict.has_key(pc):
				procDict[pc] = len(procDict)
		diagList = [diagDict[dc] for dc in diagCat]
		procList = [procDict[pc] for pc in hcpcsCat]
		if claimId == row[claimIdx]:
			## same claim means same patient, so just add
			claimDiag.extend(diagList)
			claimHcpcs.extend(procList)
			continue
		if claimId != None:
			## otherwise claim is different - so store off the old claim
			if len(claimDiag) > 0 and len(claimHcpcs) > 0:
				dpCombo = extmath.cartesian((claimDiag, claimHcpcs))
				pid = patDict[patientId]
				tensorIdx = np.append(tensorIdx, np.column_stack((np.repeat(pid, dpCombo.shape[0]), dpCombo)), axis=0)
				tensorVal = np.append(tensorVal, np.ones((dpCombo.shape[0], 1), dtype=np.int), axis=0)
		## now we juse just update the new patient
		patientId = row[pidIdx]
		claimId = row[claimIdx]
		if not patDict.has_key(patientId):
			patDict[patientId] = len(patDict)
		claimDiag = diagList
		claimHcpcs = procList
		pid += 1
	tensorIdx = np.delete(tensorIdx, (0), axis=0)
	tensorVal = np.delete(tensorVal, (0), axis=0)
	tenX = sptensor.sptensor(tensorIdx, tensorVal, np.array([len(patDict), len(diagDict), len(procDict)]))
	axisDict = {0: patDict, 1: diagDict, 2: procDict}
	return tenX, axisDict


########### PARSE THE BENEFICIARY FILE FOR CLASS INFO #############
PID_IDX = 0
CARRIER_AMT_IDX = 29
CARRIER_CUTOFF = 1640 + 1980 + 1300

patStatus = []

def getChronicDisease(line):
	patChron = np.array(line)[CHRON_IDX]
	patChron = patChron.astype(np.int)
	patChron = np.remainder(patChron, 2)
	return patChron

def parseBeneficiary(f, patClass):
	headerRow = True
	patientId = None
	patInfo = None
	for row in csv.reader(open(f, "rb")):
		if headerRow:
			headerRow = False
			continue
		if patClass.has_key(row[PID_IDX]):
			patClass[row[PID_IDX]] = patClass[row[PID_IDX]] + float(row[CARRIER_AMT_IDX])
	return patClass

tenX, axisDict = parseCarrier("hDE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv")
patDict = axisDict[0]
patClass = OrderedDict(zip(patDict.keys(), list(itertools.repeat(0, len(patDict.keys())))))
benClass = parseBeneficiary("hDE1_0_2008_to_2010_Beneficiary_Summary_File_Sample_1.csv", patClass)
patClassification = OrderedDict(zip(patDict.keys(), [v > CARRIER_CUTOFF for v in patClass.values()]))

tensorTools.saveSingleTensor(tenX, axisDict, patClassification, "cms-tensor-{0}.dat")
