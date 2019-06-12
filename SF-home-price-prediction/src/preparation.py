import os
import pandas as pd
import numpy as np
import csv
import urllib.request
import json

'''
df_full_ipo_date_97_to_19_silicon_valley - This is the data that I scraped from the NASDAQ website that lists all the
IPO's in Silicon Valley betnween 1997 and 2019/04. I will need to add updated IPO results if any new IPO's are announced
but for right now this is static --> REAL TIME WAY TO UPDATE WHEN NEW IPOS ARE ANNOUNCED
zip_all_census_data_silicon_valley - census data from all cities and beaureaus -- that holds econometric stats but also 
race, sex, and age stats about each zipcode
df_all_home_by_zip_silicon_valley - All home prices from 1996 to 2019
'''

def load_real_estate_data():

	'''
	data processing to only view california real estate data

	df_median_all_home_price_by_zip = pd.read_csv('data/raw/Zip_MedianListingPrice_AllHomes.csv',  encoding="ISO-8859-1")
	df_single_family_residence_by_zip = pd.read_csv('data/raw/Zip_Zhvi_SingleFamilyResidence.csv',  encoding="ISO-8859-1")

	df_median_all_home_price_by_zip_california = df_median_all_home_price_by_zip[df_median_all_home_price_by_zip['State']=='CA']
	df_single_family_residence_by_zip_california = df_single_family_residence_by_zip.loc[df_single_family_residence_by_zip['State']=='CA']
	'''
	df_sale_counts_by_zip = pd.read_csv('data/raw/Sale_Counts_Zip.csv', encoding="ISO-8859-1")
	df_all_home_by_zip = pd.read_csv('data/raw/Zip_Zhvi_AllHomes.csv',  encoding="ISO-8859-1")

	df_sale_counts_by_zip_california = df_sale_counts_by_zip.loc[df_sale_counts_by_zip['StateName']=='California']
	df_all_home_by_zip_california = df_all_home_by_zip.loc[df_all_home_by_zip['State']=='CA']

def load_IPO_data():
	# data processing to load all IPO Data between 1997 and present data. This data has been scraped using code from src/web_scrapers.py
	df_full_ipo_date_97_to_19 = pd.read_csv('data/processed/1997-04_2019_full_ipo_data.csv',encoding="ISO-8859-1")

	df_ritter_ipo_data_97_to_19 = pd.read_csv('data/raw/ipo_ritter_data.csv',encoding="ISO-8859-1")


def load_census_data():
	'''
	data processing of census data to create demographics by zip code
	'''
	zip_census_data = pd.read_csv('data/raw/zip_census_bureau_economic_characteristics_2017.csv',encoding="ISO-8859-1")
	zip_census_age_race_data = pd.read_csv('data/raw/zip_census_bureau_age_race_2017.csv',encoding="ISO-8859-1")

def add_zipcodes(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        your_list = list(reader)
        for z in range(0, len(your_list)):
            zipcodes[your_list[z][0]]= [],[]
            zip_list.append(your_list[z][0])

def load_zipcode_data():
	'''
	creating list and dictionary for storing all the zipcodes in silicon valley + san Francisco
	'''

	zipcodes = {} # structured with within 5, 10 miles from another zipcode
	zip_list=[]

	with open('data/raw/Santa_Clara_County_Zipcodes.csv', 'r', encoding='utf-8-sig') as f:
	    reader = csv.reader(f)
	    your_list = list(reader)
	for z in range(0, len(your_list)):
	    zipcodes[your_list[z][0]]= [],[]
	    zip_list.append(your_list[z][0])
	    
	with open('data/raw/San_Mateo_County_Zipcodes.csv', 'r', encoding='utf-8-sig') as f:
	    reader = csv.reader(f)
	    your_list = list(reader)
	for z in range(0, len(your_list)):
	    zipcodes[your_list[z][0]]= [],[]
	    zip_list.append(your_list[z][0])
	    
	with open('data/raw/San_Francisco_County_Zipcodes.csv', 'r', encoding='utf-8-sig') as f:
	    reader = csv.reader(f)
	    your_list = list(reader)
	for z in range(0, len(your_list)):
	    zipcodes[your_list[z][0]]= [],[]
	    zip_list.append(your_list[z][0])
	    
	with open('data/raw/Alameda_County_Zipcodes.csv', 'r', encoding='utf-8-sig') as f:
	    reader = csv.reader(f)
	    your_list = list(reader)
	for z in range(0, len(your_list)):
	    zipcodes[your_list[z][0]]= [],[]
	    zip_list.append(your_list[z][0])

	return zipcodes, ziplist

def wrangle_real_estate_data():
	'''
	#type wrangling of all pandas series in each dataframe

	Dataframes not being used
	df_single_family_residence_by_zip_silicon_valley = df_single_family_residence_by_zip_california[df_single_family_residence_by_zip_california['RegionName'].isin(zip_list)]
	df_median_all_home_price_by_zip_silicon_valley = df_median_all_home_price_by_zip_california[df_median_all_home_price_by_zip_california['RegionName'].isin(zip_list)]
	
	'''

	df_sale_counts_by_zip_california["RegionID"] = df_sale_counts_by_zip_california["RegionID"].astype(dtype=np.int64)
	df_sale_counts_by_zip_california['StateName'] = df_sale_counts_by_zip_california['StateName'].astype('str')

	df_sale_counts_by_zip_silicon_valley = df_sale_counts_by_zip_california[df_sale_counts_by_zip_california['RegionName'].isin(zip_list)]
	df_all_home_by_zip_silicon_valley = df_all_home_by_zip_california[df_all_home_by_zip_california['RegionName'].isin(zip_list)]

	df_all_home_by_zip_silicon_valley = df_all_home_by_zip_silicon_valley.drop(['City', 'RegionID','State','Metro','CountyName', 'SizeRank'], axis=1)
	df_sale_counts_by_zip_silicon_valley = df_sale_counts_by_zip_silicon_valley.drop(['RegionID','StateName','SizeRank', 'seasAdj'], axis=1)

def wrangle_IPO_data():
	df_full_ipo_date_97_to_19['Date Filed'] =  pd.to_datetime(df_full_ipo_date_97_to_19['Date Filed'], errors='coerce', format='%m/%d/%Y')
	df_full_ipo_date_97_to_19['Lockup Expiration Date'] =  pd.to_datetime(df_full_ipo_date_97_to_19['Lockup Expiration Date'], errors='coerce', format='%m/%d/%Y')

	df_full_ipo_date_97_to_19_silicon_valley = df_full_ipo_date_97_to_19[df_full_ipo_date_97_to_19['Zipcode'].isin(zip_list)]

	df_full_ipo_date_97_to_19_silicon_valley = df_full_ipo_date_97_to_19_silicon_valley.drop(['Lockup Expiration Date', 'Lockup Period'], axis=1)

	# causing warning -- issues with grabbing the 180, but all lockup periods here are 180
	df_full_ipo_date_97_to_19_silicon_valley['Lockup Expiration Date'] = df_full_ipo_date_97_to_19_silicon_valley['Date Filed']+ timedelta(days=180)

def wrangle_census_data():
	'''
	data wrangling of census data
	'''

	zip_census_data.rename(columns = {'Id2':'Zipcode'}, inplace=True)
	zip_census_data.rename(columns = {'Percent; EMPLOYMENT STATUS - Civilian labor force - Unemployment Rate':'Unemployment Rate'}, inplace=True)
	zip_census_data.rename(columns = {'Percent; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Less than $10,000':'l10000'}, inplace=True)
	zip_census_data.rename(columns = {'Percent; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - $10,000 to $14,999':'l15000'}, inplace=True)
	zip_census_data.rename(columns = {'Percent; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - $15,000 to $24,999':'l25000'}, inplace=True)
	zip_census_data.rename(columns = {'Estimate; COMMUTING TO WORK - Mean travel time to work (minutes)':'Mean Travel Time to Work Estimate (minutes)'}, inplace=True)
	zip_census_data.rename(columns = {'Percent; INCOME AND BENEFITS (IN 2017| INFLATION-ADJUSTED DOLLARS) - Total households - $200,000 or more':'Percent of Households with Income Greater than $200,000'}, inplace=True)
	zip_census_data.rename(columns = {'Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)':'Median Household Income Estimate (dollars)'}, inplace=True)
	zip_census_data.rename(columns = {'Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Mean household income (dollars)':'Mean Household Income Estimate (dollars)'}, inplace=True)
	zip_census_data.rename(columns = {'Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Per capita income (dollars)':'Per Capita Income Estimate (dollars)'}, inplace=True)
	zip_census_data.rename(columns = {'Percent; HEALTH INSURANCE COVERAGE - Civilian noninstitutionalized population - No health insurance coverage':'Percent of Population with no Health Insurance Coverage'}, inplace=True)
	zip_census_data.rename(columns = {'Percent; PERCENTAGE OF FAMILIES AND PEOPLE WHOSE INCOME IN THE PAST 12 MONTHS IS BELOW THE POVERTY LEVEL - All people':'Percent of People whose Income in the Past 12 months has been Below Poverty Level'}, inplace=True)


	filtered_census_data_columns = ['Zipcode',
	                               'Unemployment Rate',
	                               'Mean Travel Time to Work Estimate (minutes)',
	                               'Percent of Households with Income Greater than $200,000',
	                               'Median Household Income Estimate (dollars)',
	                               'Mean Household Income Estimate (dollars)',
	                               'Per Capita Income Estimate (dollars)',
	                               'Percent of Population with no Health Insurance Coverage',
	                               'Percent of People whose Income in the Past 12 months has been Below Poverty Level',
	                               'Percent of Households With Income Less Than $24,999']
	zip_census_data['l10000'].replace("-","0.0", regex=True, inplace=True)
	zip_census_data['l10000'].replace("N","0.0", regex=True, inplace=True)
	zip_census_data['l10000'] = zip_census_data['l10000'].astype(float)

	zip_census_data['l15000'].replace("-","0.0", regex=True, inplace=True)
	zip_census_data['l15000'].replace("N","0.0", regex=True, inplace=True)
	zip_census_data['l15000'] = zip_census_data['l15000'].astype(float)

	zip_census_data['l25000'].replace("-","0.0", regex=True, inplace=True)
	zip_census_data['l25000'].replace("N","0.0", regex=True, inplace=True)
	zip_census_data['l25000'] = zip_census_data['l25000'].astype(float)
	zip_census_data["Percent of Households With Income Less Than $24,999"] = zip_census_data['l10000'] + zip_census_data['l15000']+zip_census_data['l25000']
	zip_census_data = zip_census_data.filter(filtered_census_data_columns)

	zip_census_age_race_data.rename(columns = {'Id2':'Zipcode'}, inplace=True)
	zip_census_age_race_data.rename(columns = {'Estimate; SEX AND AGE - Median age (years)':'Median Age'}, inplace=True)
	zip_census_age_race_data.rename(columns = {'Percent; SEX AND AGE - Under 18 years':'Percent of People under 18 years of age'}, inplace=True)
	zip_census_age_race_data.rename(columns = {'Percent; SEX AND AGE - 65 years and over':'Percent of People 65 years and over'}, inplace=True)
	zip_census_age_race_data.rename(columns = {'Percent; SEX AND AGE - 18 years and over - Male':'Percent of Males'}, inplace=True)
	zip_census_age_race_data.rename(columns = {'Percent; SEX AND AGE - 18 years and over - Female':'Percent of Females'}, inplace=True)
	zip_census_age_race_data.rename(columns = {'Percent; HISPANIC OR LATINO AND RACE - Total population - Hispanic or Latino (of any race)':'Percent of People who are Hispanic'}, inplace=True)
	zip_census_age_race_data.rename(columns = {'Percent; HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - White alone':'Percent of People who are White'}, inplace=True)
	zip_census_age_race_data.rename(columns = {'Percent; HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - Black or African American alone':'Percent of People who are Black or African American'}, inplace=True)
	zip_census_age_race_data.rename(columns = {'Percent; HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - Asian alone':'Percent of People who are Asian'}, inplace=True)

	filtered_census_age_race_data_columns = ['Zipcode',
	                               'Median Age',
	                               'Percent of People under 18 years of age',
	                               'Percent of People 65 years and over',
	                               'Percent of Males',
	                               'Percent of Females',
	                               'Percent of People who are Hispanic',
	                               'Percent of People who are White',
	                               'Percent of People who are Black or African American',
	                               'Percent of People who are Asian']
	zip_census_age_race_data = zip_census_age_race_data.filter(filtered_census_age_race_data_columns)

	# filter data to only Silicon Valley + San Francisco Zip Codes
	zip_census_age_race_data_silicon_valley = zip_census_age_race_data[zip_census_age_race_data['Zipcode'].isin(zip_list)]
	zip_census_data_silicon_valley = zip_census_data[zip_census_data['Zipcode'].isin(zip_list)]

def update_series_headers_real_estate():
	'''
	run before joining dataframes so keys match
	'''
	df_all_home_by_zip_silicon_valley.columns = df_all_home_by_zip_silicon_valley.columns.str.replace('All Homes ', '')
	df_sale_counts_by_zip_silicon_valley.columns = df_sale_counts_by_zip_silicon_valley.columns.str.replace('Sales Counts ', '')

	df_all_home_by_zip_silicon_valley = df_all_home_by_zip_silicon_valley.add_prefix('All Homes ')
	df_sale_counts_by_zip_silicon_valley = df_sale_counts_by_zip_silicon_valley.add_prefix('Sales Counts ')

	df_all_home_by_zip_silicon_valley.rename(columns = {'All Homes RegionName':'Zipcode'}, inplace=True)
	df_sale_counts_by_zip_silicon_valley.rename(columns = {'Sales Counts RegionName':'Zipcode'}, inplace=True)

def join_census_data():
	'''

	'''		
	zip_all_census_data_silicon_valley = zip_census_age_race_data_silicon_valley.set_index('Zipcode').join(zip_census_data_silicon_valley.set_index('Zipcode'))

def join_all_zipcode_data():
	'''

	'''

	df_zip_all_joined_data = pd.merge(zip_all_census_data_silicon_valley, df_all_home_by_zip_silicon_valley, on='Zipcode', how='inner')
	df_zip_all_joined_data = pd.merge(df_zip_all_joined_data, df_sale_counts_by_zip_silicon_valley, on='Zipcode', how='inner')

def update_series_headers_ipo_data():
	'''
	run before joining ipo data
	wrangling ritter data
	'''
	df_ritter_ipo_data_97_to_19.rename(columns = {'Ticker':'Symbol'}, inplace=True)

	df_ritter_ipo_data_97_to_19["Found"] = df_ritter_ipo_data_97_to_19["Found"].astype(dtype=np.int64)

def join_ipo_data():
	'''

	'''
	df_full_ipo_date_97_to_19_silicon_valley = df_full_ipo_date_97_to_19_silicon_valley.set_index('Symbol').join(df_ritter_ipo_data_97_to_19.set_index('Symbol'), how='left')

	df_full_ipo_date_97_to_19_silicon_valley = df_full_ipo_date_97_to_19_silicon_valley.drop(['IPO Name', 'Offer date','CUSIP','PERM'], axis=1)




def calculate_distance_between_zips(zipcode,min_radius, max_radius):
    # api-endpoint 
    URL_base = "https://api.zip-codes.com/ZipCodesAPI.svc/1.0/FindZipCodesInRadius?zipcode="
  
    URL = URL_base+zipcode+'&minimumradius='+min_radius+'&maximumradius='+max_radius+'&key=DEMOAPIKEY'
    # sending get request and saving the response as response object 
    contents = urllib.request.urlopen(URL).read()
  

    # printing the output 
    zipcodes_nearby =[]
    for i in range(1, len(json.loads(contents)['DataList'])):
        zipcodes_nearby.append(json.loads(contents)['DataList'][i]['Code'])
        
    return zipcodes_nearby

def create_zipcode_distances_dictionary(zipcodes, zip_list):
	'''
	***DONT RUN IF THESE ARE ALREADY CREATED***
	currently stored as data/processed/zipcodes_within_radius.txt
	'''
	for i in range(0, len(zip_list)):
    	zipcodes[zip_list[i]]= calculate_distance_between_zips(zip_list[i], '0','5'), calculate_distance_between_zips(zip_list[i], '5','10')
    return zipcodes

def create_text_file_from_dictionary(filename, dictionary):
	'''
	with open('data/processed/zipcodes_within_radius.txt', 'w') as json_file:
  		json.dump(zipcodes, json_file)
  	'''
  	with open(filename, 'w') as json_file:
  		json.dump(dictionary, json_file)

def export_dataframe_to_dictionary(df,name):
	'''
	export_csv = df_zip_all_joined_data.to_csv ('data/processed/df_zip_all_joined_data.csv', index = True, header=True) 
	export_csv = df_all_home_by_zip_silicon_valley.to_csv ('data/processed/df_all_home_by_zip_silicon_valley.csv', index = True, header=True)
	export_csv = zip_census_data_silicon_valley.to_csv ('data/processed/export_dataframe.csv', index = True, header=True)
	export_csv = df_full_ipo_date_97_to_19_silicon_valley.to_csv ('data/processed/df_full_ipo_date_97_to_19_silicon_valley.csv', index = True, header=True) 

	'''
	filename = 'data/processed/'+name+'.csv'
	export_csv = df.to_csv (filename, index = True, header=True) #Don't forget to add '.csv' at the end of the path

    
def filter_data_by_zipcode_list(zip_list):
	zip_census_data_silicon_valley = zip_census_data[zip_census_data['Zipcode'].isin(zip_list)]
	df_sale_counts_by_zip_silicon_valley = df_sale_counts_by_zip_california[df_sale_counts_by_zip_california['RegionName'].isin(zip_list)]
	df_all_home_by_zip_silicon_valley = df_all_home_by_zip_california[df_all_home_by_zip_california['RegionName'].isin(zip_list)]
	df_single_family_residence_by_zip_silicon_valley = df_single_family_residence_by_zip_california[df_single_family_residence_by_zip_california['RegionName'].isin(zip_list)]
	df_median_all_home_price_by_zip_silicon_valley = df_median_all_home_price_by_zip_california[df_median_all_home_price_by_zip_california['RegionName'].isin(zip_list)]




