import os

#setup env and load df's
execfile('./setup_load_data.py')

def calculateValues(TM, M):
    fms = TM.greedy_fms(M)
    fos = TM.greedy_fos(M)
    nnz = tensorTools.countTensorNNZ(M)
    return fms, fos, nnz

## load the tensor #######
loaded_X, loaded_axisDict, loaded_classDict = tensorIO.loadSingleTensor("htn-allfinite-tensor-{0}.dat")


##read in the pickles:
matrix_pkl = open("./pheno_htn_subset_analyzed_REG.pickle", "rb")
pheno_htn_subset_analyzed = pickle.load(matrix_pkl)
matrix_pkl.close()
  
matrix_pkl = open("./Yinfo_htn_subset_analyzed.pickle", "rb")
Yinfo_all_finite = pickle.load(matrix_pkl)
matrix_pkl.close()
 

##############################################################################################################

#tensor with all phenotypes (factorization)
ktensor_phenotypes = pheno_htn_subset_analyzed
l_pts = loaded_axisDict[0].keys()
l_jdrange = loaded_axisDict[1].keys()
l_meds= loaded_axisDict[2].keys()

d_pheno_nonzero_labels = OrderedDict()

#print phenotype feature names #################
for i in range(5):
#for i in range(R):
    #print "phenotype " + str(i) + ": "
    #this_lmbda = ktensor_phenotypes.lmbda[i]
    #this_pheno_pt_factor = ktensor_phenotypes.U[0][:,i]
    #this_pheno_jdrange_factor = ktensor_phenotypes.U[1][:,i]
    #this_pheno_med_factor = ktensor_phenotypes.U[2][:,i]
    #
    #this_pheno_pt_nnz = np.nonzero(this_pheno_pt_factor)[0]
    #this_pheno_jdrange_nnz = np.nonzero(this_pheno_jdrange_factor)[0]
    #this_pheno_med_nnz = np.nonzero(this_pheno_med_factor)[0]
    #
    #l_nonzero_pt_thisPheno = []
    #l_nonzero_meds_thisPheno = []
    #l_nonzero_jdrange_thisPheno = []
    #
    #for j in this_pheno_pt_nnz:
    #    l_nonzero_pt_thisPheno.append(l_pts[j])
    #for j in this_pheno_jdrange_nnz:
    #    l_nonzero_jdrange_thisPheno.append(l_jdrange[j])
    #for j in this_pheno_med_nnz:
    #    l_nonzero_meds_thisPheno.append(l_meds[j])
    #
    ##data
    #d_pheno_nonzero_labels[i] = dict() #for phenotype i
    #d_pheno_nonzero_labels[i]['PERCENT_PTS'] = len(l_nonzero_pt_thisPheno) / float(len(this_pheno_pt_factor))
    #d_pheno_nonzero_labels[i]['MEDS_NZ'] = l_nonzero_meds_thisPheno #for phenotype i
    #d_pheno_nonzero_labels[i]['JDRANGE_NZ'] = l_nonzero_jdrange_thisPheno #for phenotype i   
    #
    #print "\tpercent pts: " + str(d_pheno_nonzero_labels[i]['PERCENT_PTS'])
    #print "\tnumber meds: " + str(len(d_pheno_nonzero_labels[i]['MEDS_NZ']))
    #print "\tnumber jdrange: " + str(len(d_pheno_nonzero_labels[i]['JDRANGE_NZ']))

    print "===== phenotype " + str(i) + "================================================================="
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
    #print "\tnumber jdrange: " + str(len(d_pheno_nonzero_labels[i]['JDRANGE_NZ']))
    #print "\tjdrange: " + str(d_pheno_nonzero_labels[i]['JDRANGE_NZ'])
    for jdrange in d_pheno_nonzero_labels[i]['JDRANGE_NZ']:
        print d_jdrange_lookup[jdrange]
    #print "\tnumber meds: " + str(len(d_pheno_nonzero_labels[i]['MEDS_NZ']))
    #print "\tmeds: " + str(d_pheno_nonzero_labels[i]['MEDS_NZ'])
    print "----------------------------------------" #divider between diagnostic codes and meds
    for med in d_pheno_nonzero_labels[i]['MEDS_NZ']:
        print med        
        
        