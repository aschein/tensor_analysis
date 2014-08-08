#do the factorization with the list of patients in question 
#
# last modified aug 6, 2014

R = 50
alpha = 1
gammaForTF = [0.001, 0.1, 0.1]



#do tensor factorization on the SUBSET - with GAMMA as set above ##################################################################################################
#laod the tensor for the subset!
loaded_X_subset_to_analyze, loaded_axisDict_subset_to_analyze, loaded_classDict_subset_to_analyze = tensorIO.loadSingleTensor("htn-tensor-subsetforanalysis-{0}.dat")

startTime = time.time()#start time -- to time it
##factorization
spntf_htn_subset_analyzed_withGamma = SP_NTF.SP_NTF(loaded_X_subset_to_analyze, R=R, alpha=alpha)
Yinfo_htn_subset_analyzed_withGamma = spntf_htn_subset_analyzed_withGamma.computeDecomp(gamma=gammaForTF)
marbleElapse = time.time() - startTime #elapsed time

#tensor decomposition factors ("phenotypes"):
pheno_htn_subset_analyzed_withGamma_REG = spntf_htn_subset_analyzed_withGamma.M[0]
pheno_htn_subset_analyzed_withGamma_AUG = spntf_htn_subset_analyzed_withGamma.M[1]
pheno_htn_subset_analyzed_withGamma = (pheno_htn_subset_analyzed_withGamma_REG, pheno_htn_subset_analyzed_withGamma_AUG) 


#string for saving the file based upon gamma
gamma_str = '_gamma'
for num in gammaForTF:
    gamma_str = gamma_str + '-' + str(num)
gamma_str = gamma_str + '.pickle'



#save factorization in pickle
outfile_str = "pheno_htn_subset_analyzed_REG" + gamma_str
with open(outfile_str, "wb") as output_file: ##IMPT! phenotype stored in this pickle
    pickle.dump(pheno_htn_subset_analyzed_withGamma, output_file)
output_file.close()
outfile_str = "pheno_htn_subset_analyzed_AUG" + gamma_str
with open(outfile_str, "wb") as output_file: ##IMPT! phenotype stored in this pickle
    pickle.dump(pheno_htn_subset_analyzed_withGamma, output_file)
output_file.close()
outfile_str = "Yinfo_htn_subset_analyzed" + gamma_str
with open(outfile_str, "wb") as output_file:
    pickle.dump(Yinfo_htn_subset_analyzed_withGamma, output_file)
output_file.close()







