#do the factorization with the list of patients in question 
#
# last modified aug 6, 2014

R = 50
alpha = 1

##classification for patients####
##classification for patients: use MAP_CHANGE < -2 as a positive change

#determine classifications of patients in the tensor
df_MAP_CHANGE_forTensorAnalysis = df_MAP_CHANGE_sample_pts
df_MAP_CHANGE_forTensorAnalysis['MAP_CHANGE_GOOD'] = df_MAP_CHANGE_forTensorAnalysis['MEDIAN_MAP_CHANGE']<=-2
df_MAP_CHANGE_forTensorAnalysis['MAP_CHANGE_GOOD'] = df_MAP_CHANGE_forTensorAnalysis['MAP_CHANGE_GOOD'].astype('int')
df_MAP_CHANGE_forTensorAnalysis = df_MAP_CHANGE_forTensorAnalysis.sort(['RUID'], ascending=1)
l_patClass_forTensorAnalysis = list(df_MAP_CHANGE_forTensorAnalysis['MAP_CHANGE_GOOD']) #patient classifications


#take a subset of the full tensor##################################################################################################
## load the tensor #######
loaded_X, loaded_axisDict, loaded_classDict = tensorIO.loadSingleTensor("htn-allfinite-tensor-20140811-{0}.dat")

#convert to nparray
nparr_loaded_X = loaded_X.tondarray()
#patients needed: 
l_patDict_idx_patients_for_tensor = np.sort([loaded_axisDict[0][ruid] for ruid in l_sample_pts])

nparr_subset_for_analysis = nparr_loaded_X[l_patDict_idx_patients_for_tensor]
patDict_subset_for_analysis = OrderedDict()
for pt in l_patDict_idx_patients_for_tensor:
    patDict_subset_for_analysis[pt]=len(patDict_subset_for_analysis)
axisDict_subset_for_analysis = loaded_axisDict
axisDict_subset_for_analysis[0] = patDict_subset_for_analysis
od_patClass_subset_for_analysis = OrderedDict(zip(patDict_subset_for_analysis.keys(), l_patClass_forTensorAnalysis)) #OrderedDict of patient classifications

# build SPARSE tensor for subset for analysis 
num_dims = len(nparr_subset_for_analysis.shape)
nnz = np.nonzero(nparr_subset_for_analysis)
data_values = nparr_subset_for_analysis[nnz].flatten()
data_values = np.reshape(data_values, (len(data_values), 1))
nonzero_subs = np.zeros((len(data_values), num_dims))
nonzero_subs.dtype = 'int'
for n in range(num_dims):
    nonzero_subs[:, n] = nnz[n]
sparse_tensor_subset_for_analysis = sptensor.sptensor(nonzero_subs, data_values)

#save tensor subset for analysis
tensorIO.saveSingleTensor(sparse_tensor_subset_for_analysis, axisDict_subset_for_analysis, od_patClass_subset_for_analysis, "htn-tensor-subsetforanalysis-20140811-{0}.dat") #

