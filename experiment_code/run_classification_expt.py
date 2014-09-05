##
## run classification using the tensor factors as features
## Sept 5, 2014
##


## prerequisites

## prerequisites

import os
import sys
import operator
import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import cross_validation 
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc

tensor_input = '/nv/pcoc1/rchen87/tensor_factorization/github_tensor/htn_data_tensors/htn-tensor-subsetforanalysis-20140811-{0}.dat'
CODE_DIR = '/nv/pcoc1/rchen87/tensor_factorization/github_tensor/experiment_code/'
marble_output_folder = '/nv/pcoc1/rchen87/tensor_factorization/github_tensor/htn_expt_run/marble_output_files/'
save_folder = '/nv/pcoc1/rchen87/tensor_factorization/github_tensor/htn_expt_run/analyzeTensors_runClassification/'
mht_feature_csvfile = '/nv/pcoc1/rchen87/download_from_dropbox_ANALYSIS_FULL_DATASET/data/new_data_20140416/Data_curated_RC_2014may/df_BPSTATUS_Phenotype_BMI_ECG_EGFR_BPCHANGE.csv'
l_mht_patients_file = '/nv/pcoc1/rchen87/tensor_factorization/github_tensor/htn_data_process/l_pts_used_MHT_outcome_analysis.txt'


#load MHT subset CSV file
with open(l_mht_patients_file) as f:
    l_pts_used_MHT_outcome_analysis = f.read().splitlines()
l_pts_used_MHT_outcome_analysis = [int(n) for n in l_pts_used_MHT_outcome_analysis]
df_mht_features = pd.read_csv(mht_feature_csvfile)
df_mht_features_SUBSET = df_mht_features[df_mht_features.RUID.isin(l_pts_used_MHT_outcome_analysis)] #use THIS as the subset of MHT patients in most recently submitted JAMIA paper


#load required modules:                                                                                                                                                                                                                                                                                           
print "loading required modules"

execfile( CODE_DIR + 'setup_python_env.py')
pheWAS_xls_file = CODE_DIR + 'ICD9-2-PheWAS.xls'


#create output folder if it does not exist                                                                                                                                                                                                                                                                        
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


#load pheWAS dictionary                                                                                                                                                                                                                                                           
xls = pd.ExcelFile(pheWAS_xls_file)
df_pheWAS = xls.parse(xls.sheet_names[0])

d_jdrange_lookup = dict(zip(list(df_pheWAS.JD_X_RANGE), list(df_pheWAS.JD_X_NAME)))


############################################################################################################## 

## load the tensor #######                                                                                                                                                                                                                                                                                        
loaded_X, loaded_axisDict, loaded_classDict = tensorIO.loadSingleTensor(tensor_input)

# the following are common to all ranges of gamma
l_pts = loaded_axisDict[0].keys()
l_jdrange = loaded_axisDict[1].keys()
l_meds= loaded_axisDict[2].keys()


## load marble results -- use the ones for gamma = 0.04

#specify which gamma
l_gammas = l_gammas = [0.01 * x for x in range(1,16)]

# loop thru all gammas in the l_gammas, load the tensor factor data, and run classification
for thisgamma in l_gammas:
    
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

    #tensor with the 50 phenotypes that were computed
    ktensor_phenotypes = pheno_htn_subset_analyzed_REG_withGamma[0]

    num_pt = ktensor_phenotypes.shape[0]
    num_jdrange = ktensor_phenotypes.shape[1]
    num_med = ktensor_phenotypes.shape[2]

    #sort phenotypes by lambda values:                                                                                                                                                                                                                                            
    d_lambda_phenoNumber = OrderedDict(zip( list(range(ktensor_phenotypes.R)),
                                        list(ktensor_phenotypes.lmbda)
                                        ))
    l_phenoNumbers_sorted_by_lambda = [tup[0] for tup in sorted(d_lambda_phenoNumber.iteritems(), key=operator.itemgetter(1))][::-1]  #get a sorted list of phenotype numbers, which are sorted by using the operator.itemgetter                                                  

    #feature_matrix and target ######################################################
    feature_matrix_phenos = ktensor_phenotypes.U[0]
    feature_matrix_phenos_binary = feature_matrix_phenos.copy()
    feature_matrix_phenos_binary[feature_matrix_phenos_binary.nonzero()] = 1 #turn phenotype matrix into a binary matrix
    target_vals = np.array(loaded_classDict.values()).reshape((num_pt, 1))

    #do CV with classification with sklearn module ######################################################
    cv_folds_indexnumbers = cross_validation.KFold(num_pt, n_folds=10, shuffle=True)

    #define X, y
    X = feature_matrix_phenos_binary
    y = target_vals

    #define logreg model
    model_logistic_l2 = linear_model.LogisticRegression(penalty='l2')

    #define metrics
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    #OPEN FIGURE HANDLE
    fig = plt.figure(1)
    fig.set_size_inches(8,8)

    for train_index, test_index in cv_folds_indexnumbers:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = feature_matrix_phenos[train_index], feature_matrix_phenos[test_index]
        y_train, y_test = target_vals[train_index], target_vals[test_index]

        probas_ = model_logistic_l2.fit(X_train, y_train).predict_proba(X_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        #add plot to figure
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    #add to figure
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    #calculate mean TPR, AUC
    mean_tpr /= len(cv_folds_indexnumbers)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    #add supporting legends to figure
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: 10-fold CV Logistic Regression; MAP DECREASE BY 2mmHg; Gamma=[0.001, 0.04, 0.04]')
    plt.legend(loc="lower right")
    #save figure
    save_filename = 'htn_marble_classification_MAPDECREASEBY2' + '_gamma_' + "-".join([str(g) for g in gammaForTF_used]) + '.png'
    fig.savefig(save_folder + save_filename)





