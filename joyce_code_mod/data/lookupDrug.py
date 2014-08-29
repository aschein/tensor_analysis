'''
File to utilize the RxNorm REST API provided by NIH
'''
import requests
import difflib
import time

SEARCH_URL = "http://rxnav.nlm.nih.gov/REST/rxcui"
SUGGESTION_URL = "http://rxnav.nlm.nih.gov/REST/spellingsuggestions"
CLASS_URL = "http://rxnav.nlm.nih.gov/REST/rxcui/"
JSON_HEADER = {'Accept': 'application/json'}

SLEEP_TIME = 2

def lookupCode(drug):	
	"""
	Perform a straightforward lookup of the RXCUI
	"""	
	payload = {'name': drug, 'search': 2}
	conceptR = requests.get(SEARCH_URL, params=payload, headers=JSON_HEADER)
	crJson = conceptR.json()['idGroup']
	return crJson

def getRXCUI(drug):
    	"""
	Get the RX CUI even for misspelled drug names
	"""
	rxcui = None
	crJson = lookupCode(drug)
	if crJson.has_key('rxnormId'):
		return drug, crJson['rxnormId'][0]
	## maybe there's a mis-spelling so find the closest one
	payload = {'name': drug}
	suggestR = requests.get(SUGGESTION_URL, params=payload, headers=JSON_HEADER)
	suggestJson = suggestR.json()
	## if there's no suggestion can't figure out what it is
	if suggestJson['suggestionGroup']['suggestionList'] == None:
		return drug, None
	## make sure the new suggestion is close at least
	newDrug = suggestJson['suggestionGroup']['suggestionList']['suggestion'][0]
	if difflib.SequenceMatcher(None, drug, newDrug.lower()).ratio() < 0.8 and not drug in newDrug.lower():
		return drug, None
	drug = newDrug
	crJson = lookupCode(drug)
	if crJson.has_key('rxnormId') and len(crJson['rxnormId']) > 0:
		rxcui = crJson['rxnormId'][0]
	return drug, rxcui

def convertEdgesToChildren(edges, nodeDict, rootNode):
	"""
	Convert the edges to a node containing it's own children
	Note the root node is denoted as root
	"""
	graph = {}
	for nodeId in nodeDict.keys():
		outEdges = filter(lambda x: x['nodeId1'] == nodeId, edges)
		if (nodeId == rootNode):
			graph['root'] = [nodeDict[x['nodeId2']] for x in outEdges]
		else:
			graph[nodeDict[nodeId]] = [nodeDict[x['nodeId2']] for x in outEdges]
	return graph

def getClassView(rxcui):
	"""
	Based on a drug RXnorm ID, get all the class hierarchy associated with it
	"""
	classGet = CLASS_URL + rxcui + "/hierarchy"
	payload = {'src': 'MESH' , 'type': 1}
	classR = requests.get(classGet, params=payload, headers=JSON_HEADER)
	classJson = classR.json()
	## double check to see if there's a graph actuall before returning anything
	if not (classJson['tree'].has_key('node') and classJson['tree'].has_key('edge')):
		return None
	## create a node id -> node name mapping
	nodes = classJson['tree']['node']
	nodeDict = { x['nodeId']: x['nodeName'] for x in nodes }
	edges = classJson['tree']['edge']
	rootNode = nodes[0]['nodeId']
	drugGraph = convertEdgesToChildren(edges, nodeDict, rootNode)
	return drugGraph


def getTradeName(rxcui):
	drugInfo = []
	tradeGet = CLASS_URL + rxcui + "/related"
	payload = {'src': 'rela' , 'rela': 'tradename_of'}
	tradeR = requests.get(tradeGet, params=payload, headers=JSON_HEADER)
	tradeJson = tradeR.json()
	if tradeJson.has_key('relatedGroup') and tradeJson['relatedGroup'].has_key('conceptGroup'):
		props = tradeJson['relatedGroup']['conceptGroup'][0]['conceptProperties']
		for prop in props:
		    success = True
		    drugInfo.append({"name":prop['name'], "rxcui":prop['rxcui']})
	return drugInfo

def convertToTradeName(drug):
	drugName, rxcui = getRXCUI(drug)
	if rxcui == None:
		return []
	time.sleep(SLEEP_TIME)
	drugInfo = getTradeName(rxcui)
	return drugInfo

def appendGraph(drugGraph, newGraph):
	for k, v in newGraph.items():
		if drugGraph.has_key(k):
			## append it to the graph
			drugGraph[k].extend(v)
		else:
			drugGraph[k] = v
	return drugGraph

def removeGraphDuplicates(drugGraph):
	for k, v in drugGraph.items():
		uniqueV = set(v)
		drugGraph[k] = list(uniqueV)
	return drugGraph


def getDrugCat(drug):
	drugName, rxcui = getRXCUI(drug)
	if rxcui == None:
	   return drugName, None
	time.sleep(SLEEP_TIME) ## sleep to prevent overloading
	drugGraph = getClassView(rxcui)
	if drugGraph != None:
	   return drugName, drugGraph
	time.sleep(SLEEP_TIME) ## sleep to prevent overloading
	## otherwise hope this is trademark
	success, drugInfo = getTradeName(rxcui)
	if not success:
	    return drugName, None
	time.sleep(SLEEP_TIME) ## sleep to prevent overloading
	drugGraph = {}
	for dx in drugInfo:
		tmpGraph = getClassView(dx['rxcui'])
		if tmpGraph != None:
		    drugGraph = appendGraph(drugGraph, tmpGraph)
		time.sleep(SLEEP_TIME)
	drugGraph = removeGraphDuplicates(drugGraph)
	return drugName, drugGraph