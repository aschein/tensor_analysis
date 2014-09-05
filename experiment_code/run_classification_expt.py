##
## run classification using the tensor factors as features
##
##


## prerequisites

import os
import sys

tensor_input = '/nv/pcoc1/rchen87/tensor_factorization/github_tensor/htn_data_tensors/htn-tensor-subsetforanalysis-20140811-{0}.dat'
CODE_DIR = '/nv/pcoc1/rchen87/tensor_factorization/github_tensor/experiment_code/'
marble_output_folder = '/nv/pcoc1/rchen87/tensor_factorization/github_tensor/htn_expt_run/marble_output_files/'
save_folder = '/nv/pcoc1/rchen87/tensor_factorization/github_tensor/htn_expt_run/analyzeTensors_runClassification/'

#load required modules:                                                                                                                                                                                                                                                                                           
print "loading required modules"

execfile( CODE_DIR + 'setup_python_env.py')
pheWAS_xls_file = CODE_DIR + 'ICD9-2-PheWAS.xls'


#create output folder if it does not exist                                                                                                                                                                                                                                                                        
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

############################################################################################################## 

#load pheWAS dictionary                                                                                                                                                                                                                                                                                           

xls = pd.ExcelFile(pheWAS_xls_file)
df_pheWAS = xls.parse(xls.sheet_names[0])

d_jdrange_lookup = dict(zip(list(df_pheWAS.JD_X_RANGE), list(df_pheWAS.JD_X_NAME)))


## load the tensor #######                                                                                                                                                                                                                                                                                        
loaded_X, loaded_axisDict, loaded_classDict = tensorIO.loadSingleTensor(tensor_input)

# the following are common to all ranges of gamma
l_pts = loaded_axisDict[0].keys()
l_jdrange = loaded_axisDict[1].keys()
l_meds= loaded_axisDict[2].keys()

## load marble results -- use the ones for gamma = 0.04

#specify which gamma
thisgamma = 0.04

#string for python pickle file (to read from) based upon gamma                                                                                                                                                                                                                                                                      
gammaForTF_used = [0.001, thisgamma, thisgamma]
gamma_str = '_gamma'
for num in gammaForTF_used:
    gamma_str = gamma_str + '-' + str(num)
gamma_str = gamma_str + '.pickle'

filename_tensorFactors_thisgamma_REG = marble_output_folder + "pheno_htn_subset_analyzed_REG" + gamma_str
filename_tensorFactors_thisgamma_AUG = marble_output_folder + "pheno_htn_subset_analyzed_AUG" + gamma_str
filename_Yinfo_thisgamma = marble_output_folder + "Yinfo_htn_subset_analyzed" + gamma_str

##read in the pickles:                                                                                                                                                                                                                                                                                            
matrix_pkl = open(filename_tensorFactors_thisgamma_REG, "rb")
pheno_htn_subset_analyzed_REG_withGamma = pickle.load(matrix_pkl)
matrix_pkl.close()

matrix_pkl = open(filename_tensorFactors_thisgamma_AUG, "rb")
pheno_htn_subset_analyzed_AUG_withGamma = pickle.load(matrix_pkl)
matrix_pkl.close()

matrix_pkl = open(filename_Yinfo_thisgamma, "rb")
Yinfo_htn_subset_analyzed_withGamma = pickle.load(matrix_pkl)
matrix_pkl.close()


############################################################################################################
#
# classification
#
#


