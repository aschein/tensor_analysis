'''
File to utilize the RxNorm REST API provided by 

http://www.nlm.nih.gov/medlineplus/druginfo/herb_All.html
'''
import requests

SEARCH_URL = "http://rxnav.nlm.nih.gov/REST/rxcui"
SUGGESTION_URL = "http://rxnav.nlm.nih.gov/REST/spellingsuggestions"
CLASS_URL = "http://rxnav.nlm.nih.gov/REST/rxcui/"
JSON_HEADER = {'Accept': 'application/json'}


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
		rxcui = crJson['rxnormId'][0]
	else:
		## maybe there's a mis-spelling so find the closest one
		payload = {'name': drug}
		suggestR = requests.get(SUGGESTION_URL, params=payload, headers=JSON_HEADER)
		suggestJson = suggestR.json()
		## if there's no suggestion can't figure out what it is
		if suggestJson['suggestionGroup']['suggestionList'] != None:
			drug = suggestJson['suggestionGroup']['suggestionList']['suggestion'][0]
			crJson = lookupCode(drug)
			if len(crJson['rxnormId']) > 0:
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
	drugName = None
	drugRxcui = None
	tradeGet = CLASS_URL + rxcui + "/related"
	payload = {'src': 'rela' , 'rela': 'tradename_of'}
	tradeR = requests.get(tradeGet, params=payload, headers=JSON_HEADER)
	tradeJson = tradeR.json()
	if tradeJson.has_key('relatedGroup'):
		props = tradeJson['relatedGroup']['conceptGroup'][0]['conceptProperties']
		drugName = props[0]['name']
		drugRxcui = props[0]['rxcui']
	return drugName, drugRxcui

def getDrugCat(drug):
	drugName, rxcui = getRXCUI(drug)
	if rxcui == None:
		return drugName, None
	drugGraph = getClassView(rxcui)
	if drugGraph != None:
		return drugName, drugGraph
	## otherwise hope this is trademark
	drugName, rxcui = getTradeName(rxcui)
	drugGraph = getClassView(rxcui)
	return drugName, drugGraph