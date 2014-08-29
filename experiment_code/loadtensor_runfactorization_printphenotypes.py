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
##     gammaForTF: numbers separated by commas, ex: '0.001, 0.1, 0.1' 
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
## pre-requisites: load environment -- these scripts need to be run first:
##
## 
###########################################################################################

## sample inputs:
#R = 50
#alpha = 1
#gammaForTF = [0.001, 0.1, 0.1]
#save_folder = './pickle_folder_20140828/'
#tensor_input = "htn-tensor-subsetforanalysis-20140811-{0}.dat"

import sys
R = int(sys.argv[1])
alpha = float(sys.argv[2])
gammaForTF = sys.argv[3]
tensor_input = sys.argv[4]
CODE_DIR = sys.argv[5]
save_folder = sys.argv[6]

gammaForTF = gammaForTF.split(',')
gammaForTF = [float(x) for x in gammaForTF]


#load required modules:
print "loading required modules"

execfile( CODE_DIR + 'setup_python_env.py')
pheWAS_xls_file = CODE_DIR + 'ICD9-2-PheWAS.xls'


#create output folder if it does not exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#do tensor factorization on the SUBSET - with GAMMA as set above ##################################################################################################
#laod the tensor for the subset!
print "loading tensor data"

loaded_X, loaded_axisDict, loaded_classDict = tensorIO.loadSingleTensor(tensor_input)

startTime = time.time()#start time -- to time it
##factorization
print "running factorization"
spntf_htn_subset_analyzed_withGamma = SP_NTF.SP_NTF(loaded_X, R=R, alpha=alpha)
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


###########################################################################################################################
##
## now, load the pickle'd phenotypes, convert to readable phenotype format, and print into output file!
##
##


import operator

def calculateValues(TM, M):
    fms = TM.greedy_fms(M)
    fos = TM.greedy_fos(M)
    nnz = tensorTools.countTensorNNZ(M)
    return fms, fos, nnz

## load the tensor #######
loaded_X, loaded_axisDict, loaded_classDict = tensorIO.loadSingleTensor(tensor_input)

##read in the pickles:

outfile_str = save_folder + "pheno_htn_subset_analyzed_REG" + gamma_str
matrix_pkl = open(outfile_str, "rb")
pheno_htn_subset_analyzed_REG_withGamma = pickle.load(matrix_pkl)
matrix_pkl.close()

outfile_str = save_folder + "Yinfo_htn_subset_analyzed" + gamma_str  
matrix_pkl = open(outfile_str, "rb")
Yinfo_htn_subset_analyzed_withGamma = pickle.load(matrix_pkl)
matrix_pkl.close()

#write output file
pheno_outstream = open(save_folder + "phenotypes"+gamma_str+".out", 'w+')
    

##############################################################################################################

#load pheWAS dictionary

xls = pd.ExcelFile(pheWAS_xls_file)
df_pheWAS = xls.parse(xls.sheet_names[0])

d_jdrange_lookup = dict(zip(list(df_pheWAS.JD_X_RANGE), list(df_pheWAS.JD_X_NAME)))



#############################################################################################################


#tensor with all phenotypes (factorization)
ktensor_phenotypes = pheno_htn_subset_analyzed_REG_withGamma[0]
l_pts = loaded_axisDict[0].keys()
l_jdrange = loaded_axisDict[1].keys()
l_meds= loaded_axisDict[2].keys()

#will store all the data
d_pheno_nonzero_labels = OrderedDict()


#sort phenotypes by lambda values:
d_lambda_phenoNumber = OrderedDict(zip( list(range(R)), 
                                        list(ktensor_phenotypes.lmbda)
                                        ))
l_phenoNumbers_sorted_by_lambda = [tup[0] for tup in sorted(d_lambda_phenoNumber.iteritems(), key=operator.itemgetter(1))][::-1]  #get a sorted list of phenotype numbers, which are sorted by using the operator.itemgetter                                

#print phenotype feature names #################
#for i in range(10):
for i in l_phenoNumbers_sorted_by_lambda:
    print "===== phenotype " + str(i) + "================================================================="
    pheno_outstream.write("===== phenotype " + str(i) + "=================================================================" + '\n')
    this_lmbda = ktensor_phenotypes.lmbda[i]
    this_pheno_pt_factor = ktensor_phenotypes.U[0][:,i]
    this_pheno_jdrange_factor = ktensor_phenotypes.U[1][:,i]
    this_pheno_med_factor = ktensor_phenotypes.U[2][:,i]
    
    this_pheno_pt_nnz = np.nonzero(this_pheno_pt_factor)[0]
    this_pheno_jdrange_nnz = np.nonzero(this_pheno_jdrange_factor)[0]
    this_pheno_med_nnz = np.nonzero(this_pheno_med_factor)[0]
    
    l_nonzero_pt_thisPheno = []
    l_nonzero_meds_thisPheno = []
    l_nonzero_jdrange_thisPheno = []
    l_nonzero_jdrange_names_thisPheno = []
    
    for j in this_pheno_pt_nnz:
        l_nonzero_pt_thisPheno.append(l_pts[j])
    for j in this_pheno_jdrange_nnz:
        l_nonzero_jdrange_thisPheno.append(l_jdrange[j])
        l_nonzero_jdrange_names_thisPheno.append(d_jdrange_lookup[l_jdrange[j]])
    for j in this_pheno_med_nnz:
        l_nonzero_meds_thisPheno.append(l_meds[j])
    
    #data
    d_pheno_nonzero_labels[i] = dict() #for phenotype i
    d_pheno_nonzero_labels[i]['LAMBDA'] = this_lmbda #lambda value
    d_pheno_nonzero_labels[i]['PERCENT_PTS'] = len(l_nonzero_pt_thisPheno) / float(len(this_pheno_pt_factor))
    d_pheno_nonzero_labels[i]['MEDS_NZ'] = l_nonzero_meds_thisPheno #for phenotype i
    d_pheno_nonzero_labels[i]['JDRANGE_NZ'] = l_nonzero_jdrange_thisPheno #for phenotype i   
    d_pheno_nonzero_labels[i]['JDRANGE_NAMES_NZ'] = l_nonzero_jdrange_names_thisPheno #for phenotype i 
    
    print "proportion of pts: " + str(d_pheno_nonzero_labels[i]['PERCENT_PTS'])
    pheno_outstream.write("proportion of pts: " + str(d_pheno_nonzero_labels[i]['PERCENT_PTS']) + '\n')
    print "lambda: " + str(this_lmbda)
    pheno_outstream.write("lambda: " + str(this_lmbda) + '\n')
    

    print "----------------------------------------" #divider
    pheno_outstream.write("----------------------------------------" + '\n')
    #make ranking of JDRANGE by the weights:
    nparr_jdrange_weights = this_pheno_jdrange_factor[this_pheno_jdrange_nnz]
    d_jdrangeindex_weights = OrderedDict(zip(this_pheno_jdrange_nnz, nparr_jdrange_weights))
    l_jdrangeindex_sorted = [tup[0] for tup in sorted(d_jdrangeindex_weights.iteritems(), key=operator.itemgetter(1))][::-1] #note: use slice [::-1] to reverse list!
    for index_this_jdrange in l_jdrangeindex_sorted:
        print d_jdrange_lookup[l_jdrange[index_this_jdrange]] + '\t' + str("%.3f" %this_pheno_jdrange_factor[index_this_jdrange] )
        pheno_outstream.write(str(d_jdrange_lookup[l_jdrange[index_this_jdrange]]) + '\t' + str("%.3f" %this_pheno_jdrange_factor[index_this_jdrange])  +'\n')
    

    print "----------------------------------------" #divider between diagnostic codes and meds
    pheno_outstream.write("----------------------------------------" + '\n')
    #make ranking of MED by the weights:
    nparr_med_weights = this_pheno_med_factor[this_pheno_med_nnz]
    d_medindex_weights = OrderedDict(zip(this_pheno_med_nnz, nparr_med_weights))
    l_medindex_sorted = [tup[0] for tup in sorted(d_medindex_weights.iteritems(), key=operator.itemgetter(1))][::-1]
    for index_this_med in l_medindex_sorted:
        print l_meds[index_this_med]  + '\t' + str("%.3f" %this_pheno_med_factor[index_this_med])
        pheno_outstream.write(l_meds[index_this_med]  + '\t' + str("%.3f" %this_pheno_med_factor[index_this_med]) + '\n')
pheno_outstream.close()
        
