## Robert Chen
## Monday 5/12/2014
##
## read and process BMI file
##
##
import datetime as dt
import scipy as s

## options
input_folder = '../../data/new_data_20140416/Data_20140409/'
output_dir = '../../data/new_data_20140416/Data_curated_RC/'
pickle_dir = '../analysis_output/pickle/'

bool_make_count_matrix_ALL = 0
bool_make_count_matrix_JD_CODE = 0
bool_make_count_matrix_JD_X_RANGE = 1

## Prepare data, read data
filename = input_folder + 'ICD_9_04082014.csv'
pd.set_option('display.line_width', 300)
df_ICD = pd.read_csv(filename, sep=',')

df_ICD['EVENT_DATE'] = pd.to_datetime(df_ICD['EVENT_DATE'])

## read in the ICD excel PheWAS file
pheWAS_xls_file = input_folder + 'ICD9-2-PheWAS.xls'
xls = pd.ExcelFile(pheWAS_xls_file)
df_pheWAS = xls.parse(xls.sheet_names[0]) 

## add columns to df_ICD for pheWAS: JD_CODE and JD_X_RANGE
df_ICD = df_ICD.merge(df_pheWAS, left_on = 'ICD_9_CODE', right_on='ICD9_CODE', how = 'left' )

## make a counts matrix
if bool_make_count_matrix_ALL:
    unique_ICD_values = df_ICD.ICD_9_CODE.unique() #ARRAY with unique ICD codes as STRINGS
    df_ICD_counts = pd.DataFrame(columns=['RUID'])
    for icd in unique_ICD_values:
        if isinstance(icd, str) or isinstance(jd, unicode):
            if s.mod(len(df_ICD_counts.columns), 100) == 0:
                print len(df_ICD_counts.columns)
            df_this_icd = df_ICD[df_ICD.ICD_9_CODE==icd][['RUID', 'ICD_9_CODE']]
            df_this_icd[icd] = df_this_icd.groupby('RUID').transform('count')
            df_this_icd = df_this_icd.drop( 'ICD_9_CODE', 1)
            df_this_icd = df_this_icd.drop_duplicates()
            df_this_icd.replace(np.nan, 0)
            if len(df_ICD_counts) == 0:
                df_ICD_counts = df_this_icd.copy()
            else:
                df_ICD_counts = pd.merge(df_ICD_counts, df_this_icd, left_on='RUID', right_on='RUID', how='outer')
    df_ICD_counts.to_csv( output_dir + 'df_ICD_counts.csv', index = False)

if bool_make_count_matrix_JD_CODE:
    unique_JD_values = df_ICD.JD_CODE.unique() #ARRAY with unique ICD codes as STRINGS
    df_JD_counts = pd.DataFrame(columns=['RUID'])
    print "JD_Counts, n= " + str(len(unique_JD_values))
    for jd in unique_JD_values:
        if isinstance(jd, str) or isinstance(jd, unicode):
            if s.mod(len(df_JD_counts.columns), 100) == 0:
                print len(df_JD_counts.columns)
            df_this_jd = df_ICD[df_ICD.JD_CODE==jd][['RUID', 'JD_CODE']]
            df_this_jd[jd] = df_this_jd.groupby('RUID').transform('count')
            df_this_jd = df_this_jd.drop( 'JD_CODE', 1)
            df_this_jd = df_this_jd.drop_duplicates()
            df_this_jd.replace(np.nan, 0)
            if len(df_JD_counts) == 0: #base case
                df_JD_counts = df_this_jd.copy()
            else:
                df_JD_counts = pd.merge(df_JD_counts, df_this_jd, left_on='RUID', right_on='RUID', how='outer')
    df_JD_counts.to_csv( output_dir + 'df_JD_counts.csv', index = False)

if bool_make_count_matrix_JD_X_RANGE:
    unique_JD_X_RANGE_values = df_ICD.JD_X_RANGE.unique() #ARRAY with unique ICD codes as STRINGS
    df_JD_RANGE_counts = pd.DataFrame(columns=['RUID'])
    print "JD_X_RANGE Counts, n= " + str(len(unique_JD_X_RANGE_values))
    for jd in unique_JD_X_RANGE_values:
        if isinstance(jd, str) or isinstance(jd, unicode):
            if s.mod(len(df_JD_RANGE_counts.columns), 10) == 0:
                print len(df_JD_RANGE_counts.columns)
            df_this_jd = df_ICD[df_ICD.JD_X_RANGE==jd][['RUID', 'JD_X_RANGE', 'EVENT_DATE']]
            #drop rows where the date is before the cutoff (two years before MHT starts)
            l_two_year_cutoff = [ df_Phenotype[df_Phenotype.RUID==item]['TWO_YEAR_BEFORE_ENGAGE_DATE'].iloc[0] for item in df_this_jd['RUID']]
            df_this_jd['TWO_YEAR_BEFORE_ENGAGE_DATE'] = l_two_year_cutoff
            df_this_jd = df_this_jd[pd.to_datetime(df_this_jd.EVENT_DATE).astype(dt.datetime) > df_this_jd['TWO_YEAR_BEFORE_ENGAGE_DATE'].astype(dt.datetime)]
            #
            df_this_jd = df_this_jd.drop( 'JD_X_RANGE', 1)
            df_this_jd = df_this_jd.drop( 'EVENT_DATE', 1)
            df_this_jd = df_this_jd.drop( 'TWO_YEAR_BEFORE_ENGAGE_DATE', 1)
            series_counts_this_jd = df_this_jd.groupby('RUID').size() 
            df_this_jd_counts = pd.DataFrame({'RUID': list(series_counts_this_jd.index), jd: list(series_counts_this_jd.values)})
            df_this_jd_counts.replace(np.nan, 0)

            if len(df_JD_RANGE_counts) == 0: #base case
                df_JD_RANGE_counts = df_this_jd_counts.copy()
            else:
                df_JD_RANGE_counts = pd.merge(df_JD_RANGE_counts, df_this_jd_counts, left_on='RUID', right_on='RUID', how='outer')
    df_JD_RANGE_counts.to_csv( output_dir + 'df_JD_RANGE_counts.csv', index = False)
