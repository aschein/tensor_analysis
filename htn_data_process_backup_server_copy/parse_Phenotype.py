## Robert Chen
## Monday 5/12/2014
##
## read and process Phenotype file
## this should be one line per patient, so no need to aggregate across lines
##
import datetime as dt

def convert_dtdays_to_years(x):
    if type(x) == dt.timedelta:
        return x.days/365.
    else:
        return x

def convert_to_1_if_positive(x, pos_label):
    if x == pos_label:
        return 1
    else:
        return 0

## options
input_folder = '../../data/new_data_20140416/Data_20140409/'
output_dir = '../analysis_output/'

## Prepare data, read data
filename = input_folder + 'Phenotype_04082014.csv'
pd.set_option('display.line_width', 300)
df_Phenotype = pd.read_csv(filename)


## format dates
df_Phenotype['ENGAGE_DATE'] = pd.to_datetime(df_Phenotype['ENGAGE_DATE']).astype(dt.datetime)
df_Phenotype['ENROLL_DATE'] = pd.to_datetime(df_Phenotype['ENROLL_DATE']).astype(dt.datetime)
df_Phenotype['DOB'] = pd.to_datetime(df_Phenotype['DOB']).astype(dt.datetime)
df_Phenotype['DOD'] = pd.to_datetime(df_Phenotype['DOD']).astype(dt.datetime)

#new columns for specific races: white, black, asian, hispanic
df_Phenotype['WHITE'] = df_Phenotype['RACE'].apply(lambda x: convert_to_1_if_positive(x, 3)) # value of 3 in RACE column means white
df_Phenotype['BLACK'] = df_Phenotype['RACE'].apply(lambda x: convert_to_1_if_positive(x, 0)) # value of 0 in RACE column means black
df_Phenotype['ASIAN'] = df_Phenotype['RACE'].apply(lambda x: convert_to_1_if_positive(x, 2)) # value of 2 in RACE column means asian
df_Phenotype['HISPANIC'] = df_Phenotype['ETHNICITY'].apply(lambda x: convert_to_1_if_positive(x, 1)) # value of 2 in RACE column means asian


## new columns for age AT ENGAGE_DATE
df_Phenotype['AGE_ENGAGE'] = (df_Phenotype['ENGAGE_DATE'] - df_Phenotype['DOB']).astype(dt.timedelta)
df_Phenotype['AGE_DEATH'] = (df_Phenotype['DOD'] - df_Phenotype['DOB']).astype(dt.timedelta)
df_Phenotype['AGE_ENGAGE'] = df_Phenotype['AGE_ENGAGE'].apply(lambda x:  convert_dtdays_to_years(x) )
df_Phenotype['AGE_DEATH'] = df_Phenotype['AGE_DEATH'].apply(lambda x:  convert_dtdays_to_years(x) )

## new column for 2-year time window before EMGAGE_DATE
df_Phenotype['TWO_YEAR_BEFORE_ENGAGE_DATE'] = df_Phenotype['ENGAGE_DATE'] + dt.timedelta(-365*2)


## print other stats
counter_SEX =  Counter(list(df_Phenotype['SEX']))
counter_ETHNICITY = Counter(list(df_Phenotype['ETHNICITY']))
counter_RACE = Counter(list(df_Phenotype['RACE']))
counter_MHT_STATUS = Counter(list(df_Phenotype['MHT_STATUS']))

print "SEX counts: "
print counter_SEX
print "ETHNICITY counts: "
print counter_ETHNICITY
print "RACE counts:"
print counter_RACE
print "MHT status counts:"
print counter_MHT_STATUS
