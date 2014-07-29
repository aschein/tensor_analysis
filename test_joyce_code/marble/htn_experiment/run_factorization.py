#do the factorization with the list of patients in question 
#
#


#BP data
df_MAP_CHANGE = df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE[['RUID', 'MEDIAN_MAP_CHANGE']] #should have 6700 rows
df_MAP_CHANGE_finite = df_MAP_CHANGE[np.isfinite(df_MAP_CHANGE['MEDIAN_MAP_CHANGE'])] #should have 2845 rows
l_pts_with_MAP_CHANGE_recorded = list(df_MAP_CHANGE_finite.RUID)
##classification for patients####
##classification for patients: use MAP_CHANGE < -2 as a positive change

#determine classifications of patients in the tensor
df_MAP_CHANGE_forTensorAnalysis = df_MAP_CHANGE_finite
df_MAP_CHANGE_forTensorAnalysis['MAP_CHANGE_GOOD'] = df_MAP_CHANGE_forTensorAnalysis['MEDIAN_MAP_CHANGE']<=-2
df_MAP_CHANGE_forTensorAnalysis['MAP_CHANGE_GOOD'] = df_MAP_CHANGE_forTensorAnalysis['MAP_CHANGE_GOOD'].astype('int')
df_MAP_CHANGE_forTensorAnalysis = df_MAP_CHANGE_forTensorAnalysis.sort(['RUID'], ascending=1)
l_patClass_forTensorAnalysis = list(df_MAP_CHANGE_forTensorAnalysis['MAP_CHANGE_GOOD']) #patient classifications


#take a subset of the full tensor##################################################################################################
## load the tensor #######
loaded_X, loaded_axisDict, loaded_classDict = tensorIO.loadSingleTensor("htn-allfinite-tensor-{0}.dat")

#convert to nparray
nparr_loaded_X = loaded_X.tondarray()
#patients needed: 
l_patDict_idx_patients_for_tensor = np.sort([loaded_axisDict[0][ruid] for ruid in l_pts_with_MAP_CHANGE_recorded])

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
tensorIO.saveSingleTensor(sparse_tensor_subset_for_analysis, axisDict_subset_for_analysis, od_patClass_subset_for_analysis, "htn-tensor-subsetforanalysis-{0}.dat") #


#do tensor factorization on the SUBSET - No GAMMA!##################################################################################################
#laod the tensor for the subset!
loaded_X_subset_to_analyze, loaded_axisDict_subset_to_analyze, loaded_classDict_subset_to_analyze = tensorIO.loadSingleTensor("htn-tensor-subsetforanalysis-{0}.dat")

startTime = time.time()#start time -- to time it
##factorization
spntf_htn_subset_analyzed = SP_NTF.SP_NTF(loaded_X_subset_to_analyze, R=R, alpha=alpha)
Yinfo_htn_subset_analyzed = spntf_htn_subset_analyzed.computeDecomp()
marbleElapse = time.time() - startTime #elapsed time

#tensor decomposition factors ("phenotypes"):
pheno_htn_subset_analyzed_REG = spntf_htn_subset_analyzed.M[0]
pheno_htn_subset_analyzed_AUG = spntf_htn_subset_analyzed.M[1]


#save factorization in pickle
with open("pheno_htn_subset_analyzed.pickle", "wb") as output_file: ##IMPT! phenotype stored in this pickle
    pickle.dump(pheno_htn_subset_analyzed, output_file)
output_file.close()
with open("Yinfo_htn_subset_analyzed.pickle", "wb") as output_file:
    pickle.dump(Yinfo_htn_subset_analyzed, output_file)
output_file.close()
with open("spntf_htn_subset_analyzed.pickle", "wb") as output_file:
    pickle.dump(spntf_htn_subset_analyzed, output_file)
output_file.close()



#do tensor factorization on the SUBSET - with GAMMA=[0.0001, 0.01, 0.01] ##################################################################################################
#laod the tensor for the subset!
loaded_X_subset_to_analyze, loaded_axisDict_subset_to_analyze, loaded_classDict_subset_to_analyze = tensorIO.loadSingleTensor("htn-tensor-subsetforanalysis-{0}.dat")

startTime = time.time()#start time -- to time it
##factorization
spntf_htn_subset_analyzed_withGamma = SP_NTF.SP_NTF(loaded_X_subset_to_analyze, R=R, alpha=alpha)
Yinfo_htn_subset_analyzed_withGamma = spntf_htn_subset_analyzed.computeDecomp(gamma=[0.0001, 0.1, 0.01])
marbleElapse = time.time() - startTime #elapsed time

#tensor decomposition factors ("phenotypes"):
pheno_htn_subset_analyzed_withGamma_REG = spntf_htn_subset_analyzed_withGamma.M[0]
pheno_htn_subset_analyzed_withGamma_AUG = spntf_htn_subset_analyzed_withGamma.M[1]
pheno_htn_subset_analyzed_withGamma = (pheno_htn_subset_analyzed_withGamma_REG, pheno_htn_subset_analyzed_withGamma_AUG) 

#save factorization in pickle
with open("pheno_htn_subset_analyzed_withGamma.pickle", "wb") as output_file: ##IMPT! phenotype stored in this pickle
    pickle.dump(pheno_htn_subset_analyzed_withGamma, output_file)
output_file.close()
with open("Yinfo_htn_subset_analyzed_withGamma.pickle", "wb") as output_file:
    pickle.dump(Yinfo_htn_subset_analyzed_withGamma, output_file)
output_file.close()
with open("spntf_htn_subset_analyzed_withGamma.pickle", "wb") as output_file:
    pickle.dump(spntf_htn_subset_analyzed_withGamma, output_file)
output_file.close()







