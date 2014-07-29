## test code on small dataset

import pandas as pd
import os
import numpy as np
import random
import json
import argparse
import time
import sys
from collections import OrderedDict
sys.path.append("..")


#operating dir
operating_dir = '/Users/localadmin/tensor_factorization/test_joyce_code/marble'
os.chdir(operating_dir)


import tensor
import SP_NTF
import simultTools
import sptensor #need this to build sparse tensor
import tensorIO # need this to save tensor 
import tensorTools

#set data dirs
data_dir = '/Users/localadmin/HTN_Predictive/data/new_data_20140416/Data_curated_RC/'
file_med = data_dir + 'df_MEDS_HTN_counts.csv'
file_jd = data_dir + 'df_JD_counts.csv'
file_jdrange = data_dir + 'df_JD_RANGE_counts.csv'
file_df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE = data_dir + 'df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE.csv'


#set properties
exptID = 3
R = 10
alpha = 1
MSize = [2,2,2]
gamma = None
AFill = [2,2,2]
startSeed = 1
outerIter = 1
innerIter = 10

#import the data
df_med = pd.read_csv(file_med)
df_jd = pd.read_csv(file_jd)
df_jdrange = pd.read_csv(file_jdrange)
df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE = pd.read_csv(file_df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE)

#take a sample of the whole dataset
df_MAP_CHANGE = df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE[['RUID', 'MEDIAN_MAP_CHANGE']] #should have 6700 rows
df_MAP_CHANGE_finite = df_MAP_CHANGE[np.isfinite(df_MAP_CHANGE['MEDIAN_MAP_CHANGE'])] #should have 2845 rows

#subset that we want (first 10 RUID's where the MAP_CHANGE was not nan)
first_10_ruid = np.sort(pd.unique(df_MAP_CHANGE_finite['RUID'])[0:10])

#grab parts of df's corredsponding to RUID we want (first 10)
df_med_first_10_ruid = df_med[df_med.RUID.isin(first_10_ruid)]
df_jd_first_10_ruid = df_jd[df_jd['RUID'].isin(first_10_ruid)]
df_jdrange_first_10_ruid = df_jdrange[df_jdrange['RUID'].isin(first_10_ruid)]
df_MAP_CHANGE_first_10_ruid = df_MAP_CHANGE_finite[df_MAP_CHANGE_finite['RUID'].isin(first_10_ruid)]

#re-sort the df's so that they are sorted by RUID (all of them should be ordered the same way by RUID)
df_med_first_10_ruid = df_med_first_10_ruid.sort(['RUID'], ascending=1)
df_jd_first_10_ruid = df_jd_first_10_ruid.sort(['RUID'], ascending=1)
df_jdrange_first_10_ruid = df_jdrange_first_10_ruid.sort(['RUID'], ascending=1)
df_MAP_CHANGE_first_10_ruid = df_MAP_CHANGE_first_10_ruid.sort(['RUID'], ascending=1)


#convert to np arrays
df_med_cols = [col for col in df_med_first_10_ruid if col not in ['RUID']] #column names for meds
df_jd_cols = [col for col in df_jd_first_10_ruid if col not in ['RUID']] #column names for jd codes
df_jdrange_cols = [col for col in df_jdrange_first_10_ruid if col not in ['RUID']] #column names for jd ranges
df_med_test_data = df_med_first_10_ruid[df_med_cols]
df_jd_test_data = df_jd_first_10_ruid[df_jd_cols]
df_jdrange_test_data = df_jdrange_first_10_ruid[df_jdrange_cols]
nparr_med_test_data = np.array(df_med_test_data)
nparr_jd_test_data = np.array(df_jd_test_data)
nparr_jdrange_test_data = np.array(df_jdrange_test_data)
#convert nan to 0
jd_nans = np.isnan(nparr_jd_test_data)
nparr_jd_test_data[jd_nans] = 0
jdrange_nans = np.isnan(nparr_jdrange_test_data)
nparr_jdrange_test_data[jdrange_nans] = 0

num_med = len(df_med_cols)
num_jd = len(df_jd_cols)
num_jdrange = len(df_jdrange_cols)

##build tensor - just do binary for now

#start with list of lists
d_ruid_matrixMedJd = dict()
l_data_by_pt = []
l_data_pt_med_jdrange = []
for ruid_index in range(len(first_10_ruid)):
    reshape_med_this_ruid = np.reshape(nparr_med_test_data[ruid_index,:], [len(nparr_med_test_data[0,:]), 1])
    jd_this_ruid = nparr_jd_test_data[ruid_index,:]
    jdrange_this_ruid = nparr_jdrange_test_data[ruid_index,:]
    matrix_hasboth_med_jd_this_ruid = reshape_med_this_ruid * jd_this_ruid  #for med*jd; if positive, this ruid has both the med and jd  
    matrix_hasboth_med_jdrange_this_ruid = reshape_med_this_ruid * jdrange_this_ruid #for med*jdrange;
    nnz_jd = np.nonzero(matrix_hasboth_med_jd_this_ruid) #for med*jd nonzero indexes
    nnz_jdrange = np.nonzero(matrix_hasboth_med_jdrange_this_ruid) #for med*jdrange; nonzero indexes
    matrix_hasboth_med_jd_this_ruid[nnz_jd] = 1  
    matrix_hasboth_med_jdrange_this_ruid[nnz_jdrange] = 1
    matrix_hasboth_med_jd_this_ruid = matrix_hasboth_med_jd_this_ruid.tolist()
    matrix_hasboth_med_jdrange_this_ruid = matrix_hasboth_med_jdrange_this_ruid.tolist()
    l_data_by_pt.append(list(matrix_hasboth_med_jd_this_ruid))
    l_data_pt_med_jdrange.append(list(matrix_hasboth_med_jdrange_this_ruid))
    
#build axisDict
patDict = OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, patient mode
medDict =  OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, med mode
jdDict = OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, jd mode
jdrangeDict = OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, jdrange mode
for pt in first_10_ruid:
    patDict[pt] = len(patDict)
for med in df_med_cols: 
    medDict[med] = len(medDict)
for jdrange in df_jdrange_cols:
    jdrangeDict[jdrange] = len(jdrangeDict)
axisDict = {0: patDict, 1: medDict, 2: jdrangeDict}

#classification for patients: use MAP_CHANGE < -2 as a positive change
df_MAP_CHANGE_first_10_ruid['MAP_CHANGE_GOOD'] = df_MAP_CHANGE_first_10_ruid['MEDIAN_MAP_CHANGE']<=-2
df_MAP_CHANGE_first_10_ruid['MAP_CHANGE_GOOD'] = df_MAP_CHANGE_first_10_ruid['MAP_CHANGE_GOOD'].astype('int')
l_patClass = df_MAP_CHANGE_first_10_ruid['MAP_CHANGE_GOOD']
od_patClass_first_10_ruid = OrderedDict(zip(patDict.keys(), l_patClass))


# build SPARSE tensor from our data
nparr_data_by_pt = np.array(l_data_pt_med_jdrange)
num_dims = len(nparr_data_by_pt.shape)
nnz = np.nonzero(nparr_data_by_pt)
data_values = nparr_data_by_pt[nnz].flatten()
data_values = np.reshape(data_values, (len(data_values), 1))
nonzero_subs = np.zeros((len(data_values), num_dims))
nonzero_subs.dtype = 'int'
for n in range(num_dims):
    nonzero_subs[:, n] = nnz[n]
sparse_tensor_first_10_ruid = sptensor.sptensor(nonzero_subs, data_values)


#save the tensor
tensorIO.saveSingleTensor(sparse_tensor_first_10_ruid, axisDict, od_patClass_first_10_ruid, "htn-first10-tensor-{0}.dat") #

### LEFT OFF HERE: june 25, 6pm ##################################################################

## load the tensor #######
loaded_X, loaded_axisDict, loaded_classDict = tensorIO.loadSingleTensor("htn-first10-tensor-{0}.dat")

## do the decomposition ######
#store the data in "data"
data = {'exptID': exptID, 'size': MSize, 'sparsity': AFill, "rank": R, "alpha": alpha, "gamma": gamma}

def calculateValues(TM, M):
    fms = TM.greedy_fms(M)
    fos = TM.greedy_fos(M)
    nnz = tensorTools.countTensorNNZ(M)
    return fms, fos, nnz
##raw features
#rawFeatures = predictionTools.createRawFeatures(X)
startTime = time.time()#start time -- to time it
##factorization
spntf_htn_first_10_ruid = SP_NTF.SP_NTF(loaded_X, R=R, alpha=alpha)
Yinfo_first_10_ruid = spntf_htn_first_10_ruid.computeDecomp()

marbleElapse = time.time() - startTime #elapsed time
#marbleFMS, marbleFOS, marbleNNZ = calculateValues(loaded_X, spntf_htn_first_10_ruid.M[SP_NTF.REG_LOCATION]) #this is wrong - fix it

with open('results/htn-first10-result-{0}.json'.format(exptID), 'w') as outfile:
    json.dump(data, outfile)                



##run *test_code_htn_analyzeFactors_rc.py*





### LEFT OFF HERE: july 8, 10pm ###########################################################################################

##bias tensor; calculate from MFact
#pftMat_htn_first_10_ruid, pftBias_htn_first_10_ruid = spntf_htn_first_10_ruid.projectData(spntf_htn_first_10_ruid, 0, maxinner=innerIter)
#	
### store off the raw file
#MFact[0].writeRawFile("results/pred-htn-first10-marble-{0}.dat".format(exptID))
#MFact[1].writeRawFile("results/pred-htn-first10-bias-marble-{0}.dat".format(exptID))
#
#
#### do the decomposition ###################################################################
##data = {'exptID': exptID, 'size': MSize, 'sparsity': AFill, 'sample': sample,
##		"rank": R, "alpha": alpha, "gamma": gamma, "seed": startSeed}
#### set seed for consistency
##seed = startSeed
##np.random.seed(seed)
##startTime = time.time()
##spntf = SP_NTF.SP_NTF(X_samp_sparse, R) #input the SPARSE representation of tensor --> does NTF!
##Yinfo = spntf.computeDecomp(gamma=gamma)
##totalTime = time.time() - startTime



				