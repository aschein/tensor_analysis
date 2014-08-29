'''
Scrape a list of supplements and herbs
'''
import urllib2
from bs4 import BeautifulSoup
import json
import string

def scrapeNIH():
	"""
	Function to scrape MedlinePlus Herbs & Supplements Page:
	http://www.nlm.nih.gov/medlineplus/druginfo/herb_All.html
	"""
	supplements = []

	PAGE_URL = "http://www.nlm.nih.gov/medlineplus/druginfo/herb_All.html"
	soup = BeautifulSoup(urllib2.urlopen(PAGE_URL).read())
	ulList = soup.find_all('ul', 'herbul')

	for ul in ulList:
		for li in ul.findAll('li'):
			supplements.append(li.find('a').getText().lower())
			print li.find('a').getText()
	supplements = list(set(supplements))
	return supplements

def scrapeRXList():
	"""
	Function to scrape rxlist for their classified supplements
	"""
	supplementDict = {}
	PAGE_URLS = ["http://www.rxlist.com/supplements/alpha_"+i+".html" for i in string.lowercase]
	for page in PAGE_URLS:
		print "Scraping page:" + str(page)
		soup = BeautifulSoup(urllib2.urlopen(page).read())
		contentMaterial = soup.find_all('div', 'contentstyle')
		for li in contentMaterial[0].findAll('li'):
			txt = li.find('a').getText() + ' '
			## try to encode it in ascii
			txt = txt.encode('ascii', 'ignore').lower()
			suppClass = str(txt)
			if txt.find("("):
				suppClass = txt[txt.rfind("(")+1:txt.find(")")]
				txt = txt[:txt.find("(")].strip()
			supplementDict[txt] = suppClass
	## make sure all the values are keys themselves
	vals = supplementDict.values()
	valDict = zip(vals, vals)
	supplementDict.update(valDict)
	return supplementDict

def main():
	supplements = scrapeRXList()
	with open('supplement.json', 'wb') as outfile:
		json.dump(supplements, outfile)

if __name__ == "__main__":
    main()