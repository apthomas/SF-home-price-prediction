import os
import pandas as pd
import numpy as np
import csv
import urllib.request
import json


def load_real_estate_data():
	 '''

	data processing to only view california real estate data from zillow.com/data
	'''
	df_sale_counts_by_zip = pd.read_csv('data/raw/Sale_Counts_Zip.csv', encoding="ISO-8859-1")
	df_median_all_home_price_by_zip = pd.read_csv('data/raw/Zip_MedianListingPrice_AllHomes.csv',  encoding="ISO-8859-1")
	df_all_home_by_zip = pd.read_csv('data/raw/Zip_Zhvi_AllHomes.csv',  encoding="ISO-8859-1")
	df_single_family_residence_by_zip = pd.read_csv('data/raw/Zip_Zhvi_SingleFamilyResidence.csv',  encoding="ISO-8859-1")

	df_sale_counts_by_zip_california = df_sale_counts_by_zip.loc[df_sale_counts_by_zip['StateName']=='California']
	df_all_home_by_zip_california = df_all_home_by_zip.loc[df_all_home_by_zip['State']=='CA']
	df_single_family_residence_by_zip_california = df_single_family_residence_by_zip.loc[df_single_family_residence_by_zip['State']=='CA']
	df_median_all_home_price_by_zip_california = df_median_all_home_price_by_zip[df_median_all_home_price_by_zip['State']=='CA']

def load_IPO_data():
	'''
	# data processing to load all IPO Data between 1997 and present data. This data has been scraped using code from src/web_scrapers.py

	'''
	df_full_ipo_date_97_to_19 = pd.read_csv('data/processed/1997-04_2019_full_ipo_data.csv',encoding="ISO-8859-1")


def load_census_data():
	'''
	data processing of census data to create demographics by zip code
	'''
	zip_census_data = pd.read_csv('data/raw/zip_census_bureau_economic_characteristics_2017.csv',encoding="ISO-8859-1")

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
	'''
	df_sale_counts_by_zip_california["RegionID"] = df_sale_counts_by_zip_california["RegionID"].astype(dtype=np.int64)
	df_sale_counts_by_zip_california['StateName'] = df_sale_counts_by_zip_california['StateName'].astype('str')

def wrangle_IPO_data():
	df_full_ipo_date_97_to_19['Date Filed'] =  pd.to_datetime(df_full_ipo_date_97_to_19['Date Filed'], errors='coerce', format='%m/%d/%Y')
	df_full_ipo_date_97_to_19['Lockup Expiration Date'] =  pd.to_datetime(df_full_ipo_date_97_to_19['Lockup Expiration Date'], errors='coerce', format='%m/%d/%Y')

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
	zip_census_data.filter(filtered_census_data_columns)



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
    
def filter_data_by_zipcode_list(zip_list):
	zip_census_data_silicon_valley = zip_census_data[zip_census_data['Zipcode'].isin(zip_list)]
	df_sale_counts_by_zip_silicon_valley = df_sale_counts_by_zip_california[df_sale_counts_by_zip_california['RegionName'].isin(zip_list)]
	df_all_home_by_zip_silicon_valley = df_all_home_by_zip_california[df_all_home_by_zip_california['RegionName'].isin(zip_list)]
	df_single_family_residence_by_zip_silicon_valley = df_single_family_residence_by_zip_california[df_single_family_residence_by_zip_california['RegionName'].isin(zip_list)]
	df_median_all_home_price_by_zip_silicon_valley = df_median_all_home_price_by_zip_california[df_median_all_home_price_by_zip_california['RegionName'].isin(zip_list)]



