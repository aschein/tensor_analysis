#if on local mac machine, do this:
#

# load these each time have to reload python
setup_python_env.py
read_pheWAS_dictionary.py
resume_reloadTensors_localadmin.py

# if initially doing it
parse_MEDS_MHT_subset_localadmin.py
build_sparse_tensor_from_mht_data.py

#jump to these if data has been parsed before
run_factorization.py
create_subset_of_allfinite_tensor_forAnalysis.py
analyze_tensor_factors.py #if gamma used: analyze_tensor_factors_withGamma.py


