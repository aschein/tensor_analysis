import sys
import operator

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

############################################################################################################## 

##
## For rest of the script, do the following:
##    loop through all iterations where gamma was different, and grab phenotypes information from those
##

## load the tensor #######                                                                                                                                                                                                                                                                                        
loaded_X, loaded_axisDict, loaded_classDict = tensorIO.loadSingleTensor(tensor_input)

# the following are common to all ranges of gamma
l_pts = loaded_axisDict[0].keys()
l_jdrange = loaded_axisDict[1].keys()
l_meds= loaded_axisDict[2].keys()

# count the number of factors for both diagnoses / meds
#   -- note: this is assuming we used same gamma for diagnoses as well as meds
#
l_gammas_used = [x*0.01 for x in range(1,16)]

d_numfactors_pt_perGamma = dict()
d_numfactors_diag_perGamma = dict()
d_numfactors_med_perGamma = dict()

d_numPheno_nonzero_pt_perGamma = dict()
d_numPheno_nonzero_diag_perGamma = dict()
d_numPheno_nonzero_med_perGamma = dict()

for thisgamma in l_gammas_used:
    #string for saving the file based upon gamma                                                                                                                                                                                                                                                                      
    gammaForTF_used = [0.001, thisgamma, thisgamma]
    gamma_str = '_gamma'
    for num in gammaForTF_used:
        gamma_str = gamma_str + '-' + str(num)
    gamma_str = gamma_str + '.pickle'
    
    #filename for this set of gammas
    filename_tensorFactors_thisgamma = marble_output_folder + "pheno_htn_subset_analyzed_REG" + gamma_str
    filename_Yinfo_thisgammae = marble_output_folder + "Yinfo_htn_subset_analyzed" + gamma_str
    
    ##read in the pickles:                                                                                                                                                                                                                                                                                            
    matrix_pkl = open(filename_tensorFactors_thisgamma, "rb")
    pheno_htn_subset_analyzed_REG_withGamma = pickle.load(matrix_pkl)
    matrix_pkl.close()

    matrix_pkl = open(filename_Yinfo_thisgammae, "rb")
    Yinfo_htn_subset_analyzed_withGamma = pickle.load(matrix_pkl)
    matrix_pkl.close()

    #tensor with all phenotypes (factorization)                                                                                                                                                                                                                                                                       
    ktensor_phenotypes = pheno_htn_subset_analyzed_REG_withGamma[0]

    #sort phenotypes by lambda values:                                                                                                                                                                                                                                                                                
    d_lambda_phenoNumber = OrderedDict(zip( list(range(ktensor_phenotypes.R)),
                                        list(ktensor_phenotypes.lmbda)
                                        ))
    l_phenoNumbers_sorted_by_lambda = [tup[0] for tup in sorted(d_lambda_phenoNumber.iteritems(), key=operator.itemgetter(1))][::-1]  #get a sorted list of phenotype numbers, which are sorted by using the operator.itemgetter                                                                                      
    
    # loop through phenotypes and count how many diagnosis / meds there are for each phenotype
    l_numfactors_pt = []
    l_numfactors_jdrange = []
    l_numfactors_med = []
    for i in l_phenoNumbers_sorted_by_lambda:
        this_pheno_pt_factor = ktensor_phenotypes.U[0][:,i]
        this_pheno_jdrange_factor = ktensor_phenotypes.U[1][:,i]
        this_pheno_med_factor = ktensor_phenotypes.U[2][:,i]
    
        this_pheno_pt_nnz = np.nonzero(this_pheno_pt_factor)[0]
        this_pheno_jdrange_nnz = np.nonzero(this_pheno_jdrange_factor)[0]
        this_pheno_med_nnz = np.nonzero(this_pheno_med_factor)[0]
        
        num_pt = len(this_pheno_pt_nnz)
        num_jdrange = len(this_pheno_jdrange_nnz)
        num_med = len(this_pheno_med_nnz)
    
        l_numfactors_pt.append(num_pt)
        l_numfactors_jdrange.append(num_jdrange)
        l_numfactors_med.append(num_med)
    
    d_numfactors_pt_perGamma[thisgamma] = np.array(l_numfactors_pt)[np.mean(np.nonzero(l_numfactors_pt))]
    d_numfactors_diag_perGamma[thisgamma] = np.array(l_numfactors_jdrange)[np.mean(np.nonzero(l_numfactors_jdrange))]
    d_numfactors_med_perGamma[thisgamma] = np.array(l_numfactors_med)[np.mean(np.nonzero(l_numfactors_med))]
    d_numPheno_nonzero_pt_perGamma[thisgamma] = len(np.nonzero(l_numfactors_pt))
    d_numPheno_nonzero_diag_perGamma[thisgamma] = len(np.nonzero(l_numfactors_jdrange))
    d_numPheno_nonzero_med_perGamma[thisgamma] = len(np.nonzero(l_numfactors_med))
        
od_numfactors_pt_perGamma = OrderedDict(sorted(d_numfactors_pt_perGamma.items()))
od_numfactors_diag_perGamma = OrderedDict(sorted(d_numfactors_diag_perGamma.items()))
od_numfactors_med_perGamma = OrderedDict(sorted(d_numfactors_med_perGamma.items()))
od_numPheno_nonzero_pt_perGamma = OrderedDict(sorted(d_numPheno_nonzero_pt_perGamma.items()))
od_numPheno_nonzero_diag_perGamma = OrderedDict(sorted(d_numPheno_nonzero_diag_perGamma.items()))
od_numPheno_nonzero_med_perGamma = OrderedDict(sorted(d_numPheno_nonzero_med_perGamma.items()))


#make plots

fig = plt.figure(1)
fig.set_size_inches(8,8)
plt.plot(od_numfactors_pt_perGamma.keys(), od_numfactors_pt_perGamma.values() , 'b')
plt.xlabel('gamma value')
plt.ylabel('Count')
plt.title('Patient factor')
fig.savefig(save_folder + 'htn_marble_ptMode_gamma_factorElementCount.png')
plt.close()

fig = plt.figure(2)
fig.set_size_inches(8,8)
plt.plot(od_numfactors_diag_perGamma.keys(), od_numfactors_diag_perGamma.values() , 'b')
plt.xlabel('gamma value')
plt.ylabel('Count')
plt.title('Diagnosis factor')
fig.savefig(save_folder + 'htn_marble_diagMode_gamma_factorElementCount.png')
plt.close()

fig = plt.figure(3)
fig.set_size_inches(8,8)
plt.plot(od_numfactors_med_perGamma.keys(), od_numfactors_med_perGamma.values() , 'b')
plt.xlabel('gamma value')
plt.ylabel('Count')
plt.title('Medication factor')
fig.savefig(save_folder + 'htn_marble_medMode_gamma_factorElementCount.png')
plt.close()

