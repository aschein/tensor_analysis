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

if bool_initial_run == 0:
    pickfile = open('./d_meds_classes.pickle', 'rb')
    d_meds_classes = pickle.load(pickfile)
    pickfile.close()
    
    pickfile2 = open('./nparr_pt_jdrange_med_first699.pickle', 'rb')
    nparr_pt_jdrange_med_first699 = pickle.load(pickfile2)
    pickfile2.close()
    
    resume_pt_index = 699
    
if compiled_drug_dictionary == 0:
    execfile('./lookupDrug.py')

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
    
    
    ## load phenotype file
    fdsafdas 

    ## Prepare data, read data
    print("preparing and loading data ..........")
    read_filename = input_folder + 'Meds_DD_04082014_withHeader.csv'
    store_filename = input_folder + "df_MEDS.h5"
    jdrange_filename = output_dir + "df_jdrange_all_entries.csv"
    df_allmeds_mht_filename = output_dir + "df_MEDS_ALLMEDS_MHT.csv"
#    file_df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE = input_curated_dir + 'df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE.csv'
    file_df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE = '../../data/df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE.csv'   
    
    df_jdrange_allentries = pd.read_csv(jdrange_filename)
    l_jdrange_names_unique = list(np.sort(df_jdrange_allentries.JD_X_RANGE.unique()))
    df_MEDS_ALLMEDS_MHT = pd.read_csv(df_allmeds_mht_filename)
    df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE = pd.read_csv(file_df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE)
    
    l_med_classes_unique = list(np.sort(d_meds_classes.values()))


#
### read in med classes file
#medclasses_xls = pd.ExcelFile(file_classes)
#df_medclasses = medclasses_xls.parse(medclasses_xls.sheet_names[0])
##build dictionary with HTN med classes:
#d_drug_classes = dict()
#d_drug_classes_by_name = dict()
#for ind in range(len(df_medclasses['Hypertension_Med_Classes'])):
#    key = str(df_medclasses['Hypertension_Med_Classes'][ind]).upper()
#    val_drug = str(df_medclasses['Drug_Name'][ind]).upper()
#    val_brand = str(df_medclasses['Brand_Name'][ind]).upper()
#    if key in d_drug_classes.keys():
#        d_drug_classes[key].append(val_drug)
#        d_drug_classes[key].append(val_brand)
#    else:
#        d_drug_classes[key] = list()
#        d_drug_classes[key].append(val_drug)
#        d_drug_classes[key].append(val_brand)
#
#for ind in range(len(df_medclasses['Drug_Name'])):
#    key1 = str(df_medclasses['Drug_Name'][ind]).upper()
#    key2 = str(df_medclasses['Brand_Name'][ind]).upper()
#    value = str(df_medclasses['Hypertension_Med_Classes'][ind]).upper()
#    if key1 not in d_drug_classes_by_name.keys():
#        d_drug_classes_by_name[key1] = value
#    if key2 not in d_drug_classes_by_name.keys():
#        d_drug_classes_by_name[key2] = value
#



#########################################################################################################################################################
#
##rxNorm conversion
#execfile('./lookupDrug.py') #load the functions for looking up drug class in rxNorm



#
### read the med data in chunks; its 11M lines - too big to read the whole thing in memory
#cnt = 0
#print "reading chunks, ~11.2M lines total; 500K-line chunks; 23 chunks total"  
#for chunk in pd.read_csv(read_filename, chunksize=500000, escapechar='\\'):
#    cnt = cnt + 1
#    print "start chunk number: " + str(cnt)
#
#    #split names for which there is a colon separating generic name and brand name
#    l_med_leftofcolon = [str(item).split(':')[0].strip().upper() for item in chunk['Drug_Name']]
#    l_med_rightofcolon = [str(item).split(':')[1].strip().upper() if len(str(item).split(':'))>1 else str(item).upper() for item in chunk['Drug_Name']]
#    chunk['DRUG_NAME_GENERIC'] = l_med_leftofcolon
#    chunk['DRUG_NAME_BRAND'] = l_med_rightofcolon
#    
#    #check if patient is in the MHT subset
#    df_MEDS_this_chunk_in_MHT_subset = chunk[chunk['RUID'].isin(l_pts_used_MHT_outcome_analysis)]
#
#    #check if they are HTN meds
#    df_MEDS_this_chunk_are_htn_meds = df_MEDS_this_chunk_in_MHT_subset[df_MEDS_this_chunk_in_MHT_subset['DRUG_NAME_GENERIC'].isin(d_drug_classes_by_name.keys())]
#    l_classes = [d_drug_classes_by_name[item] for item in df_MEDS_this_chunk_are_htn_meds['DRUG_NAME_GENERIC']]
#    df_MEDS_this_chunk_are_htn_meds['DRUG_CLASS'] = l_classes
#
#    #append the HTN meds ,ALLMEDS
#    df_MEDS_HTN = df_MEDS_HTN.append(df_MEDS_this_chunk_are_htn_meds)
#    df_MEDS_ALLMEDS = df_MEDS_ALLMEDS.append(df_MEDS_this_chunk_in_MHT_subset)
#
#df_MEDS_HTN.to_csv( output_dir + 'df_MEDS_HTN_MHT.csv', index = False)
#df_MEDS_ALLMEDS.to_csv( output_dir + 'df_MEDS_ALLMEDS.csv', index = False)
#

######build matrix for binary values ############################################################################################################

print("preparing meds data for converting to med classes / building interaction matrices ..................")

start_time = time.time()

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
for item in df_MEDS_ALLMEDS_sample_pts['DRUG_NAME_GENERIC']:
    if item in d_meds_classes:
        l_classes.append(d_meds_classes[item])
    else:
        l_classes.append("NA")
df_MEDS_ALLMEDS_sample_pts['DRUG_CLASS'] = l_classes
l_med_classes_unique = np.unique(l_classes)

#num in each dimension:
num_pts = len(l_sample_pts)
num_jdrange = len(l_jdrange_names_unique)
num_med_classes = len(l_med_classes_unique)

#indexes by name of jdrange / med
d_jdrange_index = OrderedDict()
d_med_index = OrderedDict()
for jdrange_idx in range(len(l_jdrange_names_unique)):
    jdrange_name = l_jdrange_names_unique[jdrange_idx]
    d_jdrange_index[jdrange_name] = jdrange_idx
for med_idx in range(len(l_med_classes_unique)):
    med_name = l_med_classes_unique[med_idx]
    d_med_index[med_name] = med_idx

#store the matrix
print("build the interaction matrices .......")

if bool_initial_run:
    nparr_pt_jdrange_med = np.array([])
    cnt_pt_loop = 0 #counter, for debugging purposes
else:
    nparr_pt_jdrange_med = np.copy(nparr_pt_jdrange_med_first699)

cnt_pt_loop = 699
print("number pts total = " + str(num_pts))
for pt in l_sample_pts[699:]:
    cnt_pt_loop = cnt_pt_loop + 1
    if np.mod(cnt_pt_loop, 1)== 0:
        print str(cnt_pt_loop) + "; time elapsed: " + str(time.time()-start_time)
    matrix_interaction_this_pt = np.zeros([num_jdrange, num_med_classes])
    df_jdrange_this_pt = df_jdrange_allentries[df_jdrange_allentries['RUID']==pt] #jdrange recordings
    df_med_this_pt = df_MEDS_ALLMEDS_sample_pts[df_MEDS_ALLMEDS_sample_pts['RUID']==pt]
    for entry_idx in range(len(df_jdrange_this_pt)):
        dt_jdrange = parse(df_jdrange_this_pt.iloc[entry_idx]['EVENT_DATE'])
#        dt_jdrange = dt.datetime.strptime(df_jdrange_this_pt.iloc[entry_idx]['EVENT_DATE'], "%Y-%m-%d %H:%M:%S")
        this_jdrange = df_jdrange_this_pt.iloc[entry_idx]['JD_X_RANGE'] #this JDRANGE
        for med_idx in range(len(df_med_this_pt)):
            this_jdrange_matrixidx = d_jdrange_index[this_jdrange]
            this_med = df_med_this_pt.iloc[med_idx]['DRUG_CLASS']
            this_med_matrixidx = d_med_index[this_med]
            if type(df_med_this_pt.iloc[med_idx]['Entry_Date']) == str:
                if matrix_interaction_this_pt[this_jdrange_matrixidx, this_med_matrixidx] == 0:
                    dt_med = parse(df_med_this_pt.iloc[med_idx]['Entry_Date'])
    #                dt_med = dt.datetime.strptime(df_med_this_pt.iloc[med_idx]['Entry_Date'], "%Y-%m-%d %H:%M:%S")
                    timediff = dt_jdrange - dt_med
                    if abs(timediff.days <= 7):
                        matrix_interaction_this_pt[this_jdrange_matrixidx, this_med_matrixidx] += 1
    nparr_pt_jdrange_med = np.append(nparr_pt_jdrange_med, matrix_interaction_this_pt)
nparr_pt_jdrange_med = nparr_pt_jdrange_med.reshape([num_pts, num_jdrange, num_med_classes])
make_interaction_3dmatrix_time = time.time() - start_time #elapsed time


#matrix for binary values
nnz_indexes = np.nonzero(nparr_pt_jdrange_med)
nparr_pt_jdrange_med_binary = np.copy(nparr_pt_jdrange_med)
nparr_pt_jdrange_med_binary[nnz_indexes] = 1




