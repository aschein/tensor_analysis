#script to load event data and convert to counts (interaction) tensor
import os
import sys
import dateutil
import time
from dateutil.parser import parse
if sys.version_info.major == 3:
    import pickle
else:
    import cPickle as pickle
import pandas as pd
from collections import OrderedDict

operating_dir = '/Users/localadmin/tensor_factorization/test_joyce_code/marble/htn_experiment'
os.chdir(operating_dir)

sys.path.append("..")
sys.path.append("../../")

import sptensor
import tensorIO
import SP_NTF
import tensor



#set properties
exptID = 3
R = 50
alpha = 1
gamma = None
startSeed = 1
outerIter = 1
innerIter = 10


#load the data
data_dir = '../../../data/'
med_file = data_dir + 'df_MEDS_HTN_allentries.csv'
jdrange_file = data_dir + 'df_jdrange_all_entries.csv'
file_df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE = data_dir + 'df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE.csv'

df_med_allentries = pd.read_csv(med_file)
df_jdrange_allentries = pd.read_csv(jdrange_file)
df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE = pd.read_csv(file_df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE)

#list of patients for which theres data in either meds or jdranges
l_pts_in_df_MEDS_HTN = list(df_med_allentries.RUID.unique())
l_pts_in_df_jdrange = list(df_jdrange_allentries.RUID.unique())
l_all_pts_with_MEDS_JDRANGE = list(np.unique(l_pts_in_df_MEDS_HTN + l_pts_in_df_jdrange))
#list of names of jdrange/meds; sorted in alphabetical order
l_jdrange_names_unique = list(np.sort(df_jdrange_allentries.JD_X_RANGE.unique()))
l_med_names_unique = list(np.sort(df_med_allentries.DRUG_CLASS.unique()))
#indexes by name of jdrange / med
d_jdrange_index = OrderedDict()
d_med_index = OrderedDict()
for jdrange_idx in range(len(l_jdrange_names_unique)):
    jdrange_name = l_jdrange_names_unique[jdrange_idx]
    d_jdrange_index[jdrange_name] = jdrange_idx
for med_idx in range(len(l_med_names_unique)):
    med_name = l_med_names_unique[med_idx]
    d_med_index[med_name] = med_idx

#BP data
df_MAP_CHANGE = df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE[['RUID', 'MEDIAN_MAP_CHANGE']] #should have 6700 rows
df_MAP_CHANGE_finite = df_MAP_CHANGE[np.isfinite(df_MAP_CHANGE['MEDIAN_MAP_CHANGE'])] #should have 2845 rows

#num in each dimension
num_pts = len(l_all_pts_with_MEDS_JDRANGE)
num_jdrange = len(l_jdrange_names_unique)
num_med = len(l_med_names_unique)

#now, make the interaction 3d matrix
start_time = time.time()
nparr_pt_jdrange_med = np.array([])
cnt_pt_loop = 0
#for pt in l_all_pts_with_MEDS_JDRANGE:
#for pt in l_all_pts_with_MEDS_JDRANGE[3655:]:
#for pt in l_all_pts_with_MEDS_JDRANGE[4519:]:
for pt in l_all_pts_with_MEDS_JDRANGE[4534:]:
    cnt_pt_loop = cnt_pt_loop + 1
    if mod(cnt_pt_loop, 10) == 0:
        print str(cnt_pt_loop) + "; time elapsed: " + str(time.time()-start_time)
    matrix_interaction_this_pt = np.zeros([num_jdrange, num_med])
    df_jdrange_this_pt = df_jdrange_allentries[df_jdrange_allentries.RUID==pt] #jdrange recordings
    df_med_this_pt = df_med_allentries[df_med_allentries.RUID==pt]
    for entry_idx in range(len(df_jdrange_this_pt)): #loop thru each entry
        dt_jdrange = parse(df_jdrange_this_pt.iloc[entry_idx].EVENT_DATE)
        this_jdrange = df_jdrange_this_pt.iloc[entry_idx].JD_X_RANGE #this JDRANGE
        for med_idx in range(len(df_med_this_pt)):
            dt_med = parse(df_med_this_pt.iloc[med_idx].Entry_Date)
            this_med = df_med_this_pt.iloc[med_idx].DRUG_CLASS
            timediff = dt_jdrange-dt_med
            if abs(timediff.days <= 7):
                this_jdrange_matrixidx = d_jdrange_index[this_jdrange]
                this_med_matrixidx = d_med_index[this_med]
                matrix_interaction_this_pt[this_jdrange_matrixidx, this_med_matrixidx] += 1
    nparr_pt_jdrange_med = np.append(nparr_pt_jdrange_med, matrix_interaction_this_pt)    
nparr_pt_jdrange_med = nparr_pt_jdrange_med.reshape([num_pts, num_jdrange, num_med]) #reshape by dimensions of [pt, jdrange, med]
make_interaction_3dmatrix_time = time.time() - start_time #elapsed time

#make matrix for binary values: 
nnz_indexes = np.nonzero(nparr_pt_jdrange_med) #set nonzero entries to 1
nparr_pt_jdrange_med_binary = np.copy(nparr_pt_jdrange_med)
nparr_pt_jdrange_med_binary[nnz_indexes] = 1

#save matrix?
save_matrix_filename = "./nparr_pt_jdrange_med.pickle"
with open(save_matrix_filename, "wb") as output_file:
    pickle.dump(nparr_pt_jdrange_med, output_file)
output_file.close()

##read in the pickle - if necessary:
matrix_pkl = open("./nparr_pt_jdrange_med.pickle", "rb")
nparr_pt_jdrange_med = pickle.load(matrix_pkl)
matrix_pkl.close()





##########################################################################################


# build SPARSE tensor from our data
num_dims = len(nparr_pt_jdrange_med_binary.shape)
nnz = np.nonzero(nparr_pt_jdrange_med_binary)
data_values = nparr_pt_jdrange_med_binary[nnz].flatten()
data_values = np.reshape(data_values, (len(data_values), 1))
nonzero_subs = np.zeros((len(data_values), num_dims))
nonzero_subs.dtype = 'int'
for n in range(num_dims):
    nonzero_subs[:, n] = nnz[n]
sparse_tensor_all_finite = sptensor.sptensor(nonzero_subs, data_values)



##classification for patients####
##classification for patients: use MAP_CHANGE < -2 as a positive change
#patients needed: 
l_patients_for_tensor = np.sort(list(df_MAP_CHANGE_finite.RUID))
l_patDict_idx_patients_for_tensor = np.sort([patDict[ruid] for ruid in l_patients_for_tensor])
nparr_pt_jdrange_med_binary_subset = nparr_pt_jdrange_med_binary[l_patDict_idx_patients_for_tensor]

#build axisDict
patDict = OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, patient mode
medDict =  OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, med mode
jdDict = OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, jd mode
jdrangeDict = OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, jdrange mode
for pt in l_patients_for_tensor:
    patDict[pt] = len(patDict)
for med in l_med_names_unique: 
    medDict[med] = len(medDict)
for jdrange in l_jdrange_names_unique:
    jdrangeDict[jdrange] = len(jdrangeDict)
axisDict = {0: patDict, 1: jdrangeDict, 2:medDict}


#df_MAP_CHANGE = df_MAP_CHANGE_finite[df_MAP_CHANGE_finite['RUID'].isin(l_patients_for_tensor)]
df_MAP_CHANGE['MAP_CHANGE_GOOD'] = df_MAP_CHANGE['MEDIAN_MAP_CHANGE']<=-2 
df_MAP_CHANGE['MAP_CHANGE_GOOD'] = df_MAP_CHANGE['MAP_CHANGE_GOOD'].astype('int')
df_MAP_CHANGE = df_MAP_CHANGE.sort(['RUID'], ascending=1)
l_patClass_allpts = df_MAP_CHANGE['MAP_CHANGE_GOOD'] #patient classifications
l_patClass_allfinitepts = list(df_MAP_CHANGE[df_MAP_CHANGE.RUID.isin(l_all_pts_with_MEDS_JDRANGE)]['MAP_CHANGE_GOOD'])

od_patClass_for_tensor = OrderedDict(zip(patDict.keys(), l_patClass_allfinitepts)) #OrderedDict of patient classifications

#save the tensor
tensorIO.saveSingleTensor(sparse_tensor_all_finite, axisDict, od_patClass_for_tensor, "htn-allfinite-tensor-{0}.dat") #


############################################################################################################
############################################################################################################
############################################################################################################

