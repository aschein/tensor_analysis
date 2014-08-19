## Robert Chen
## last updated Thursday 7/31/2014
##
## read and process MEDS
##
##

bool_initial_run = 1
compiled_drug_dictionary = 1

import os
import sys
os.chdir('/Users/localadmin/tensor_factorization/github_tensor/htn_data_process/')

#load the dictionary of drug classes (from json format)
import json
with open('scrape_drugs_joyce/drugDict.json') as json_drugDict:
    d_drugDict_drugsCom = json.load(json_drugDict)
with open('scrape_drugs_joyce/d_meds_classes_rxNorm.json') as json_drugDict_rxNorm:
    d_drugDict_rxNorm = json.load(json_drugDict_rxNorm)


if bool_initial_run:

    import datetime as dt
    import pandas as pd
    import dateutil
    import time
    from dateutil.parser import parse
    if sys.version_info.major == 3:
        import pickle
    else:
        import cPickle as pickle
    from collections import OrderedDict
    import json
    
    
    ## options
    input_folder = './' #note: this is a symlinkt to real data dir
    input_curated_dir = './'
    output_dir = './' #note: this is a symlink to the real data_curated_dir
    file_classes = input_folder + 'MedClasses.xlsx'
    
    with open('./l_pts_used_MHT_outcome_analysis.txt') as f:
        l_pts_used_MHT_outcome_analysis = f.read().splitlines()
    l_pts_used_MHT_outcome_analysis = [int(x) for x in l_pts_used_MHT_outcome_analysis]
    l_pts_used_MHT_outcome_analysis = np.sort(l_pts_used_MHT_outcome_analysis)
    

    ## Prepare data, read data
    print("preparing and loading data ..........")
    read_filename = input_folder + 'Meds_DD_04082014_withHeader.csv'
    store_filename = input_folder + "df_MEDS.h5"
    jdrange_filename = output_dir + "df_jdrange_all_entries.csv"
    df_allmeds_mht_filename = output_dir + "df_MEDS_ALLMEDS_MHT.csv"
#    file_df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE = input_curated_dir + 'df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE.csv'
    file_df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE = '../../data/df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE.csv'   
    #declare dataframes
    df_jdrange_allentries = pd.read_csv(jdrange_filename)
    l_jdrange_names_unique = list(np.sort(df_jdrange_allentries.JD_X_RANGE.unique()))
    df_MEDS_ALLMEDS_MHT = pd.read_csv(df_allmeds_mht_filename)
    df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE = pd.read_csv(file_df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE)
    #convert dates to datetimes
    df_jdrange_allentries['EVENT_DATE'] = pd.to_datetime(df_jdrange_allentries['EVENT_DATE']).astype(dt.datetime)
    df_MEDS_ALLMEDS_MHT['Entry_Date'] = pd.to_datetime(df_MEDS_ALLMEDS_MHT['Entry_Date']).astype(dt.datetime)
    df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE['ENGAGE_DATE'] = pd.to_datetime(df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE['ENGAGE_DATE']).astype(dt.datetime)
    

######build matrix for binary values ############################################################################################################

print("preparing meds data for converting to med classes / building interaction matrices ..................")

l_sample_pts = l_pts_used_MHT_outcome_analysis
df_MEDS_ALLMEDS_sample_pts = df_MEDS_ALLMEDS_MHT[df_MEDS_ALLMEDS_MHT.RUID.isin(l_sample_pts)]
df_BPSTATUS_sample_pts = df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE[df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE.RUID.isin(l_sample_pts)]
df_MAP_CHANGE_sample_pts = df_BPSTATUS_sample_pts[['RUID', 'MEDIAN_MAP_CHANGE']] #should only have 2521 rows (same RUID's as MHT analysis group)


#convert meds to classes
print("convert meds to classes.....")
print("form list from leftofcolon in mednames--")
l_med_leftofcolon_sample_pts = [str(item).split(':')[0].strip().upper() for item in df_MEDS_ALLMEDS_sample_pts['Drug_Name']]
print("convert meds from generic name to class name")
df_MEDS_ALLMEDS_sample_pts['DRUG_NAME_GENERIC'] = l_med_leftofcolon_sample_pts
l_classes = []
l_classes_singlemeds = []
l_classes_singlemeds_NA = [] #make a list of the drugs that cannot be mapped to a class in Drugs.com (d_drugDict_drugsCom) or rxNorm (d_drugDict_rxNorm)
cnt = 0
for item in df_MEDS_ALLMEDS_sample_pts['DRUG_NAME_GENERIC']:
    cnt +=1
    if mod(cnt, 100000) == 0:
        print cnt
    if item in d_drugDict_drugsCom:
        l_classes.append(';'.join(d_drugDict_drugsCom[item]['cat']))########CHANGE THIS!!!!!###############################################
        for single_drug in d_drugDict_drugsCom[item]['cat']:
            l_classes_singlemeds.append(single_drug)
    elif item in d_drugDict_rxNorm:
        l_classes.append(';'.join(d_drugDict_rxNorm[item])) #### CHANGE THIS!!!!! #######
        for single_drug in d_drugDict_rxNorm[item]:
            l_classes_singlemeds.append(single_drug)
    else:
        l_classes.append("NA")
        l_classes_singlemeds.append("NA")
        l_classes_singlemeds_NA.append(item)
df_MEDS_ALLMEDS_sample_pts['DRUG_CLASS'] = l_classes

l_classes_unique_singlemeds = np.unique(l_classes_singlemeds)
l_classes_unique_singlemeds = l_classes_unique_singlemeds[l_classes_unique_singlemeds != "NA"] #remove NA (NA's makeup about 7.5% of the list, as of Aug 11)


#num in each dimension:
num_pts = len(l_sample_pts)
num_jdrange = len(l_jdrange_names_unique)
num_med_classes = len(l_classes_unique_singlemeds)

#indexes by name of jdrange / med
d_jdrange_index = OrderedDict()
d_med_index = OrderedDict()
for jdrange_idx in range(len(l_jdrange_names_unique)):
    jdrange_name = l_jdrange_names_unique[jdrange_idx]
    d_jdrange_index[jdrange_name] = jdrange_idx
for med_idx in range(len(l_classes_unique_singlemeds)):
    med_name = l_classes_unique_singlemeds[med_idx]
    d_med_index[med_name] = med_idx

#store the matrix
print("build the interaction matrices .......")

#if bool_initial_run:
#    nparr_pt_jdrange_med = np.array([])
#    cnt_pt_loop = 0 #counter, for debugging purposes
#else:
#    nparr_pt_jdrange_med = np.copy(nparr_pt_jdrange_med_first699)

nparr_pt_jdrange_med = np.array([])
start_time = time.time()
cnt_pt_loop = 0
print("number pts total = " + str(num_pts))
for pt in l_sample_pts:
    cnt_pt_loop = cnt_pt_loop + 1
    if np.mod(cnt_pt_loop, 1)== 0:
        print str(cnt_pt_loop) + "; time elapsed: " + str(time.time()-start_time)
    matrix_interaction_this_pt = np.zeros([num_jdrange, num_med_classes])
    dt_engagedate_this_pt = df_BPSTATUS_sample_pts[df_BPSTATUS_sample_pts['RUID']==pt]['ENGAGE_DATE'].astype(dt.datetime).values[0]
    dt_twoyearbefore_this_pt = dt_engagedate_this_pt-dt.timedelta(730) #take 730 days = 2 years
    df_jdrange_this_pt = df_jdrange_allentries[(df_jdrange_allentries['RUID']==pt) &(df_jdrange_allentries['EVENT_DATE']>dt_twoyearbefore_this_pt)& (df_jdrange_allentries['EVENT_DATE']< dt_engagedate_this_pt)] #jdrange recordings
    df_med_this_pt = df_MEDS_ALLMEDS_sample_pts[(df_MEDS_ALLMEDS_sample_pts['RUID']==pt) & (df_MEDS_ALLMEDS_sample_pts['Entry_Date']>dt_twoyearbefore_this_pt) & (df_MEDS_ALLMEDS_sample_pts['Entry_Date']<dt_engagedate_this_pt)]
    for entry_idx in range(len(df_jdrange_this_pt)):
        dt_jdrange = df_jdrange_this_pt.iloc[entry_idx]['EVENT_DATE']
        if type(dt_jdrange) == dt.datetime:
            this_jdrange = df_jdrange_this_pt.iloc[entry_idx]['JD_X_RANGE'] #this JDRANGE
            this_jdrange_matrixidx = d_jdrange_index[this_jdrange]
            ## find dataframe of all meds entries occuring within a week of the jdrange
            df_med_within_one_week_this_pt = df_med_this_pt[(df_med_this_pt['Entry_Date']>dt_jdrange-dt.timedelta(3)) & (df_med_this_pt['Entry_Date']<dt_jdrange+dt.timedelta(3))]
            l_medclasses_within_one_week_this_pt = list(df_med_within_one_week_this_pt['DRUG_CLASS'].unique())
            for medlist in l_medclasses_within_one_week_this_pt: ##loop thru all meds that have entries within a week of the jdrange
                for med in list(medlist.split(';')):
                    if med != 'NA': #ignore the NA's!
                        this_med_matrixidx = d_med_index[med]
                        matrix_interaction_this_pt[this_jdrange_matrixidx, this_med_matrixidx] += 1       
    nparr_pt_jdrange_med = np.append(nparr_pt_jdrange_med, matrix_interaction_this_pt)
nparr_pt_jdrange_med = nparr_pt_jdrange_med.reshape([num_pts, num_jdrange, num_med_classes])
make_interaction_3dmatrix_time = time.time() - start_time #elapsed time


#matrix for binary values
nnz_indexes = np.nonzero(nparr_pt_jdrange_med)
nparr_pt_jdrange_med_binary = np.copy(nparr_pt_jdrange_med)
nparr_pt_jdrange_med_binary[nnz_indexes] = 1




