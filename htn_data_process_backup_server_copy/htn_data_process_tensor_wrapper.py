## wrapper

## set these options
workingDir = './';
bool_initial_run = 0


## import modules
execfile('./setup_python_env.py')


os.chdir(workingDir)
sys.path.append('./')

## run these scripts: ---------------------------------------------------------------------------------

if bool_initial_run:
    # read in and process data
#    execfile('newdata_dataread_BP_MHTSTRATEGY_v2.py')
    # analyze BP intervals to determine IN CONTROL / OUT OF CONTROL status
#    execfile('newdata_analyzeBP_BP_MHTSTRATEGY.py')
    #parse other files
    execfile('parse_Phenotype.py')
    execfile('parse_BMI.py')
    execfile('parse_ECG.py')
    execfile('parse_EGFR.py')
    execfile('parse_MHT_STRATEGY.py')
    execfile('parse_BP.py')
    execfile('parse_ICD.py')
    execfile('parse_CPT.py')
    execfile('parse_LAB.py')
    # make lists of patients IN control or OUT of control for HYPERTENSION set
    execfile('newdata_make_list_pts_IN_OUT.py')
    # KS test, etc for patients in the HYPERTENSION set
    execfile('./KS_2samp_test_before_after_MHT.py')
    # KS test, etc for patients in the other MHT comorbities: DIABETES and CHF sets
    execfile('./KS_2samp_test_before_after_MHT_stratComorbid.py')
    #compile master data frame
    execfile('./compile_master_dataframe.py')

if not bool_initial_run:
    print 'loading past saved data (pickles)....'
#    execfile('load_saved_pickles.py')
    print "done. "
#    execfile('parse_Phenotype.py')
#    execfile('parse_BMI.py')
#    execfile('parse_ECG.py')
#    execfile('parse_EGFR.py')
#    execfile('parse_MHT_STRATEGY.py')
#    execfile('parse_BP.py')
#    execfile('parse_ICD.py')
#    execfile('parse_CPT.py')
#    execfile('parse_LAB.py')
#    execfile('newdata_make_list_pts_IN_OUT.py')
#    execfile('./KS_2samp_test_before_after_MHT.py')
#    execfile('./KS_2samp_test_before_after_MHT_stratComorbid.py')
#    execfile('prep_data_VIP_class.py')
#    execfile('./compile_master_dataframe.py')
