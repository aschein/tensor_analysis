##after loading the HTN meds, build sparse tensor from the MHT data

#sys.path.append("../test_joyce_code/")
#import sptensor
#import tensorIO
#import SP_NTF
#import tensor


# build SPARSE tensor from our data
num_dims = len(nparr_pt_jdrange_med_binary_first699.shape)
nnz = np.nonzero(nparr_pt_jdrange_med_binary_first699)
data_values = nparr_pt_jdrange_med_binary_first699[nnz].flatten()
data_values = np.reshape(data_values, (len(data_values), 1))
nonzero_subs = np.zeros((len(data_values), num_dims))
nonzero_subs.dtype = 'int'
for n in range(num_dims):
    nonzero_subs[:, n] = nnz[n]
sparse_tensor_all_finite = sptensor.sptensor(nonzero_subs, data_values)

##classification for patients####
##classification for patients: use MAP_CHANGE < -2 as a positive change
##patients needed: 
##l_patients_for_tensor = np.sort(l_sample_pts) #list of the patients in question
l_patients_for_tensor = np.sort(l_sample_pts[0:699]) #list of the patients in question

#build axisDict
patDict = OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, patient mode
medDict =  OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, med mode
jdDict = OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, jd mode
jdrangeDict = OrderedDict(sorted({}.items(), key= lambda t:t[1])) #axis dict, jdrange mode
for pt in l_patients_for_tensor:
    patDict[pt] = len(patDict)
for med in l_med_classes_unique: 
    medDict[med] = len(medDict)
for jdrange in l_jdrange_names_unique:
    jdrangeDict[jdrange] = len(jdrangeDict)
axisDict = {0: patDict, 1: jdrangeDict, 2:medDict}

#list of patient dictionaries for tensor
l_patDict_idx_patients_for_tensor = np.sort([patDict[ruid] for ruid in l_patients_for_tensor])
##subset of nparray
#nparr_pt_jdrange_med_binary_subset = nparr_pt_jdrange_med_binary[l_patDict_idx_patients_for_tensor]
nparr_pt_jdrange_med_binary_subset = nparr_pt_jdrange_med_binary_first699[l_patDict_idx_patients_for_tensor]


#df_MAP_CHANGE = df_MAP_CHANGE_finite[df_MAP_CHANGE_finite['RUID'].isin(l_patients_for_tensor)]
df_MAP_CHANGE_subset = df_MAP_CHANGE_sample_pts.head(699)
df_MAP_CHANGE_subset['MAP_CHANGE_GOOD'] = df_MAP_CHANGE_subset['MEDIAN_MAP_CHANGE']<=-2 
df_MAP_CHANGE_subset['MAP_CHANGE_GOOD'] = df_MAP_CHANGE_subset['MAP_CHANGE_GOOD'].astype('int')
df_MAP_CHANGE_subset = df_MAP_CHANGE_subset.sort(['RUID'], ascending=1)
l_patClass_allpts = df_MAP_CHANGE_subset['MAP_CHANGE_GOOD'] #patient classifications

od_patClass_for_tensor = OrderedDict(zip(patDict.keys(), l_patClass_allpts)) #OrderedDict of patient classifications

#save the tensor
tensorIO.saveSingleTensor(sparse_tensor_all_finite, axisDict, od_patClass_for_tensor, "htn-alljdrange-allmed-first699-tensor-{0}.dat") #
