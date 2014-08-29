## this combines the 2 scripts into one comprehensive one:
##
## last modified aug 28, 2014
##
##   1. run_factorization_localadmin.py
##   2. analyze_tensor_factors_withGamma.py
##  
## INPUT:
##     R
##     alpha
##     gammaForTF
##     tensor_filename
##     save_folder
##
## OUTPUT:
##     tensor factorizaiton results in dir save_folder/
##         pheno_htn_subset_analyzed_REG_<gamma_str>.pickle
##         pheno_htn_subset_analyzed_AUG_<gamma_str>.pickle
##         Yinfo_htn_subset_analyzed_<gamma_str>.pickle
##         
##     analyzed PHENOTYPE output in save_folder/
##         phenotypes_<gamma_str>.out
##
##

##inputs here
R = 50
alpha = 1
gammaForTF = [0.001, 0.1, 0.1]
save_folder = './pickle_folder_20140828/'
tensor_input = "htn-tensor-subsetforanalysis-20140811-{0}.dat"


#create output folder if it does not exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#do tensor factorization on the SUBSET - with GAMMA as set above ##################################################################################################
#laod the tensor for the subset!
loaded_X_subset_to_analyze, loaded_axisDict_subset_to_analyze, loaded_classDict_subset_to_analyze = tensorIO.loadSingleTensor("htn-tensor-subsetforanalysis-20140811-{0}.dat")

startTime = time.time()#start time -- to time it
##factorization
print "running factorization"
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
outfile_str = save_folder + "pheno_htn_subset_analyzed_REG" + gamma_str
with open(outfile_str, "wb") as output_file: ##IMPT! phenotype stored in this pickle
    pickle.dump(pheno_htn_subset_analyzed_withGamma, output_file)
output_file.close()
outfile_str = save_folder + "pheno_htn_subset_analyzed_AUG" + gamma_str
with open(outfile_str, "wb") as output_file: ##IMPT! phenotype stored in this pickle
    pickle.dump(pheno_htn_subset_analyzed_withGamma, output_file)
output_file.close()
outfile_str = save_folder + "Yinfo_htn_subset_analyzed" + gamma_str
with open(outfile_str, "wb") as output_file:
    pickle.dump(Yinfo_htn_subset_analyzed_withGamma, output_file)
output_file.close()




