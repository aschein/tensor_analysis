#
#import os
#import json
#
#operating_dir = './'
#os.chdir(operating_dir)
#
#def calculateValues(TM, M):
#    fms = TM.greedy_fms(M)
#    fos = TM.greedy_fos(M)
#    nnz = tensorTools.countTensorNNZ(M)
#    return fms, fos, nnz
#
### load the tensor #######
#loaded_X, loaded_axisDict, loaded_classDict = tensorIO.loadSingleTensor("htn-alljdrange-allmed-first699-tensor-{0}.dat")
#
#
###read in the pickles:
#matrix_pkl = open("./pheno_htn_subset_analyzed_withGamma.pickle", "rb")
#pheno_htn_subset_analyzed_withGamma = pickle.load(matrix_pkl)
#matrix_pkl.close()
#  
#matrix_pkl = open("./Yinfo_htn_subset_analyzed_withGamma.pickle", "rb")
#Yinfo_htn_subset_analyzed_withGamma = pickle.load(matrix_pkl)
#matrix_pkl.close()
jdrange_file = open("./d_jdrange_lookup.pickle", "rb")
d_jdrange_lookup = pickle.load(jdrange_file)
jdrange_file.close()
# 

##############################################################################################################

#tensor with all phenotypes (factorization)
ktensor_phenotypes = pheno_htn_subset_analyzed_withGamma[0]
l_pts = loaded_axisDict[0].keys()
l_jdrange = loaded_axisDict[1].keys()
l_meds= loaded_axisDict[2].keys()

#will store all the data
d_pheno_nonzero_labels = OrderedDict()

#sort phenotypes by lambda values:
d_lambda_phenoNumber = OrderedDict(zip(list(pheno_htn_subset_analyzed_withGamma_REG.lmbda),
                                        list(range(R) )
                                        ))
l_phenoNumbers_sorted_by_lambda = [  d_lambda_phenoNumber[x] for x in sorted(d_lambda_phenoNumber.keys(), reverse=True)]                                  

#print phenotype feature names #################
#for i in [0]:

#to print to file, import the following
f = open('phenotypes_gamma_0p1_0p01_0p001.txt', 'a')

for i in l_phenoNumbers_sorted_by_lambda:
    print "===== phenotype " + str(i) + "================================================================="
    f.write("===== phenotype " + str(i) + "=================================================================\n")
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
    print "lambda: " + str(this_lmbda)
    print "----------------------------------------" #divider
    f.write("proportion of pts: " + str(d_pheno_nonzero_labels[i]['PERCENT_PTS']) + '\n')
    f.write("lambda: " + str(this_lmbda) + '\n')
    f.write("----------------------------------------\n")
    #print "\tnumber jdrange: " + str(len(d_pheno_nonzero_labels[i]['JDRANGE_NZ']))
    #print "\tjdrange: " + str(d_pheno_nonzero_labels[i]['JDRANGE_NZ'])
    for jdrange in d_pheno_nonzero_labels[i]['JDRANGE_NZ']:
        print d_jdrange_lookup[jdrange]
        f.write(d_jdrange_lookup[jdrange]+'\n')
    #print "\tnumber meds: " + str(len(d_pheno_nonzero_labels[i]['MEDS_NZ']))
    #print "\tmeds: " + str(d_pheno_nonzero_labels[i]['MEDS_NZ'])
    print "----------------------------------------" #divider between diagnostic codes and meds
    for med in d_pheno_nonzero_labels[i]['MEDS_NZ']:
        print med   
        f.write(med + '\n')
        
