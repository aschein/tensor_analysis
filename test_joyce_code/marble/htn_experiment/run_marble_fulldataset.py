#do tensor factorization for loaded data so far

#load pickle'd file
pickle_file = "./nparr_first4519_pt_jdrange_med.pickle"
with open(pickle_file, 'rb') as fhandle:
    nparr_first4519_pt_jdrange_med = pickle.load(fhandle)
fhandle.close()
