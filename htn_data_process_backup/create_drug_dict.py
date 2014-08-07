import pandas as pd
import os
import sys
import numpy as np
import json

execfile('./lookupDrug.py')
curated_data_dir = '../data_curated_dir/'

df_allmeds_mht_filename = curated_data_dir + "df_MEDS_ALLMEDS_MHT.csv"

df_MEDS_ALLMEDS_MHT = pd.read_csv(df_allmeds_mht_filename)

l_unique_meds = df_MEDS_ALLMEDS_MHT.DRUG_NAME_GENERIC.unique() #should have 9841 unique medication names

#define dictionary
d_meds_classes = dict()
cnt_meds = 0
for genericname in l_unique_meds:
    cnt_meds += 1
    if np.mod(cnt_meds, 100) == 0:
        print "parsing meds: " + str(cnt_meds)
    if not genericname in d_meds_classes:
        t_ping_rxnorm = getDrugCat(genericname)
        if t_ping_rxnorm[1] != None:
            d_meds_classes[genericname] = getDrugCat(genericname)[1]['root'][-1] #look up the category using the generic name; should be the LAST entry of the list!


#write dictionary to a json file
with open('d_meds_classes.json', 'wb') as fp:
    json.dump(d_meds_classes, fp)

