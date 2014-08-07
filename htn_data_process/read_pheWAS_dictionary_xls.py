#read in the dictionary of JD ranges and pheWAS codes

## read in the ICD excel PheWAS file                                                                                                                                                                                            
pheWAS_xls_file = './ICD9-2-PheWAS.xls'                                                                                                                                                                            
xls = pd.ExcelFile(pheWAS_xls_file)                                                                                                                                                                                             
df_pheWAS = xls.parse(xls.sheet_names[0]) 

d_jdrange_lookup = dict(zip(list(df_pheWAS.JD_X_RANGE), list(df_pheWAS.JD_X_NAME)))

