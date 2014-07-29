##analyze tensor factors found - for test_code_htn

num_pheno = R #number of candidate phenotypes = rank



#for each candidate phenotype: 
#   list these:
#        meds (nonzero)
#        ICD codes (nonzero)
#
#   then: find PATIENT that is MOST likely to be this one
#       list these for the PATIENT:
#           meds (nonzero)
#           ICD codes (nonzero)
#
#


#tensor with all phenotypes (factorization)
ktensor_phenotypes = spntf_htn_all_finite.M[0]
l_pts = loaded_axisDict[0].keys()
l_meds = loaded_axisDict[1].keys()
l_jdrange = loaded_axisDict[2].keys()

d_pheno_nonzero_labels = OrderedDict()

for i in range(num_pheno):
    print "phenotype " + str(i) + ": "
    this_lmbda = ktensor_phenotypes.lmbda[i]
    this_patient_factor = ktensor_phenotypes.U[0][:,i]
    this_med_factor = ktensor_phenotypes.U[1][:,i]
    this_jdrange_factor = ktensor_phenotypes.U[2][:,i]
    
    nonzero_ind_patients = np.nonzero(this_patient_factor)[0]
    nonzero_ind_med = np.nonzero(this_med_factor)[0]
    nonzero_ind_jdrange = np.nonzero(this_jdrange_factor)[0]
    
    l_nonzero_meds_thisPheno = []
    l_nonzero_jdrange_thisPheno = []
    
    for j in nonzero_ind_med:
        l_nonzero_meds_thisPheno.append(l_meds[j])
    for j in nonzero_ind_jdrange:
        l_nonzero_jdrange_thisPheno.append(l_jdrange[j])

    #data
    d_pheno_nonzero_labels[i] = dict() #for phenotype i
    d_pheno_nonzero_labels[i]['PERCENT_PTS'] = len(nonzero_ind_patients) / float(len(this_patient_factor))
    d_pheno_nonzero_labels[i]['MEDS_NZ'] = l_nonzero_meds_thisPheno #for phenotype i
    d_pheno_nonzero_labels[i]['JDRANGE_NZ'] = l_nonzero_jdrange_thisPheno #for phenotype i
    
    print "\t-percent pts: " + str(d_pheno_nonzero_labels[i]['PERCENT_PTS'])

