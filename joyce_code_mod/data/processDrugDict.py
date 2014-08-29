"""
Aggregate the drug classes a bit more, essentially choose the categories that are at most nodes down.
Example: azole antifungals is categorized using antifungals (depth of 2)

A drugDict.json file is written which corresponds to the final drugClass dictionary.
"""
import json

def aggregateDrugCat(drugTree):
    drugCat = {}
    for k, v in drugTree.iteritems():
        if len(v) < 2:
            drugCat[k] = k
        elif len(v) == 2:
            drugCat[k] = v[1]
        else:
            drugCat[k] = v[-2]
    return drugCat

def main():
    treeStruct = json.load(open('drugClass-struct.json', 'rb'))
    drugCat = aggregateDrugCat(treeStruct)
    drugClass = json.load(open('drugClass-dict.json', 'rb'))
    finalDrugClass = {}
    for k,v in drugClass.iteritems():
        v['cat'] = set([drugCat[item] for item in v['cat']])
        v['cat'] = list(v['cat'])
        finalDrugClass[k] = v
    with open('drugDict.json', 'wb') as outfile:
        json.dump(finalDrugClass, outfile, indent=2)

if __name__ == "__main__":
    main()