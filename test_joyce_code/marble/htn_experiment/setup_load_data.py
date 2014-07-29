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




