#gamma_str = '_gamma-0.001-0.01-0.01.pickle'

import operator

def calculateValues(TM, M):
    fms = TM.greedy_fms(M)
    fos = TM.greedy_fos(M)
    nnz = tensorTools.countTensorNNZ(M)
    return fms, fos, nnz

## load the tensor #######
loaded_X, loaded_axisDict, loaded_classDict = tensorIO.loadSingleTensor("htn-tensor-subsetforanalysis-20140811-{0}.dat")


##read in the pickles:
pickle_folder = "./pickle_folder_20140814/"

outfile_str = pickle_folder + "pheno_htn_subset_analyzed_REG" + gamma_str
matrix_pkl = open(outfile_str, "rb")
pheno_htn_subset_analyzed_REG_withGamma = pickle.load(matrix_pkl)
matrix_pkl.close()

outfile_str = pickle_folder + "Yinfo_htn_subset_analyzed" + gamma_str  
matrix_pkl = open(outfile_str, "rb")
Yinfo_htn_subset_analyzed_withGamma = pickle.load(matrix_pkl)
matrix_pkl.close()

#write output file
pheno_outstream = open(pickle_folder + "phenotypes"+gamma_str+".out", 'w+')
    

##############################################################################################################

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
        