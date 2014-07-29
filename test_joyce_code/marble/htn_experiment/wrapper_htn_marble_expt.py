#if haven't converted CSV events yet:
execfile('./convert_csvEvents_to_tensor.py')

#otherwise:
execfile('./setup_load_data.py')
execfile('./run_factorization.py')
execfile('./read_pheWAS_dictionary_xls.py')
execfile('./analyze_tensor_factors_withGamma.py')
