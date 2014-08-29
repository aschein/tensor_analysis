import sys
import os
os.chdir('/Users/localadmin/tensor_factorization/github_tensor/htn_data_process/')

execfile('./setup_python_env.py')
execfile('./read_pheWAS_dictionary.py')

    
#reload dict of med classes
pickfile = open('./d_meds_classes.pickle', 'rb')
d_meds_classes = pickle.load(pickfile)
pickfile.close()


## do stuff:
sys.path.append("../test_joyce_code/")
#sys.path.append("../test_joyce_code/marble")
import sptensor
import tensorIO
import SP_NTF
import tensor



