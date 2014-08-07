exptID = 3
R = 50
alpha = 1
gamma = None
startSeed = 1
outerIter = 1
innerIter  = 10


## load the tensor #######
loaded_X_subset_to_analyze, loaded_axisDict_subset_to_analyze, loaded_classDict_subset_to_analyze = tensorIO.loadSingleTensor("htn-alljdrange-allmed-first699-tensor-{0}.dat")

# build SPARSE tensor for subset for analysis 
#nparr_subset_for_analysis = loaded_X_subset_to_analyze
#num_dims = len(nparr_subset_for_analysis.shape)
#nnz = np.nonzero(nparr_subset_for_analysis)
#data_values = nparr_subset_for_analysis[nnz].flatten()
#data_values = np.reshape(data_values, (len(data_values), 1))
#nonzero_subs = np.zeros((len(data_values), num_dims))
#nonzero_subs.dtype = 'int'
#for n in range(num_dims):
#    nonzero_subs[:, n] = nnz[n]
#sparse_tensor_subset_for_analysis = sptensor.sptensor(nonzero_subs, data_values)



startTime = time.time()#start time -- to time it
##factorization
spntf_htn_subset_analyzed_withGamma = SP_NTF.SP_NTF(loaded_X_subset_to_analyze, R=R, alpha=alpha)
Yinfo_htn_subset_analyzed_withGamma = spntf_htn_subset_analyzed_withGamma.computeDecomp(gamma=[0.0001, 0.1, 0.01])
marbleElapse = time.time() - startTime #elapsed time


#tensor decomposition factors ("phenotypes"):
pheno_htn_subset_analyzed_withGamma_REG = spntf_htn_subset_analyzed_withGamma.M[0]
pheno_htn_subset_analyzed_withGamma_AUG = spntf_htn_subset_analyzed_withGamma.M[1]
pheno_htn_subset_analyzed_withGamma = (pheno_htn_subset_analyzed_withGamma_REG, pheno_htn_subset_analyzed_withGamma_AUG) 


with open("pheno_htn_subset_analyzed_withGamma.pickle", "wb") as output_file: ##IMPT! phenotype stored in this pickle
    pickle.dump(pheno_htn_subset_analyzed_withGamma, output_file)
output_file.close()
with open("Yinfo_htn_subset_analyzed_withGamma.pickle", "wb") as output_file:
    pickle.dump(Yinfo_htn_subset_analyzed_withGamma, output_file)
output_file.close()
