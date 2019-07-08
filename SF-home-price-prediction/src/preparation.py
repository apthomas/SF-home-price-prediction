import pandas as pd
import numpy as np
import csv
import urllib.request
import json
from datetime import datetime
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
import web_scrapers
import os


def load_real_estate_data(filename, state_attr, state):
    df = pd.read_csv(filename, encoding="ISO-8859-1")
    df = df.loc[df[state_attr] == state]
    return df


def load_data(filenames):
    df_list=[]
    for i in range(0, len(filenames)):
        df = pd.read_csv(filenames[i], encoding="ISO-8859-1")
        df_list.append(df)
    return df_list


def create_zipcode_list(filenames):
    zipcodes = {}  # structured with within 5, 10 miles from another zipcode
    zip_list = []
    for i in range(0, len(filenames)):
        with open(filenames[i], 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            your_list = list(reader)
            for z in range(0, len(your_list)):
                zipcodes[your_list[z][0]] = [], []
                zip_list.append(your_list[z][0])
    return zipcodes, zip_list


def wrangle_real_estate_data(df, zip_list, drop_columns):
    df = df[df['RegionName'].isin(zip_list)]
    df = df.drop(drop_columns, axis=1)
    return df


def wrangle_IPO_data(df, zip_list):
    df['Date Filed'] = pd.to_datetime(df['Date Filed'], errors='coerce', format='%m/%d/%Y')
    df['Lockup Expiration Date'] = pd.to_datetime(df['Lockup Expiration Date'], errors='coerce', format='%m/%d/%Y')
    df = df[df['Zipcode'].isin(zip_list)]
    df = df.drop(['Lockup Expiration Date', 'Lockup Period'], axis=1)
    df['Lockup Expiration Date'] = df['Date Filed'] + timedelta(days=180)
    return df


def wrangle_census_data(df_census_econ, df_census_dem, zip_list, census_econ_columns, census_dem_columns):
    df_census_econ.rename(columns={'Id2': 'Zipcode'}, inplace=True)
    df_census_econ.rename(
        columns={'Percent; EMPLOYMENT STATUS - Civilian labor force - Unemployment Rate': 'Unemployment Rate'},
        inplace=True)
    df_census_econ.rename(columns={
        'Percent; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Less than $10,000': 'l10000'},
        inplace=True)
    df_census_econ.rename(columns={
        'Percent; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - $10,000 to $14,999': 'l15000'},
        inplace=True)
    df_census_econ.rename(columns={
        'Percent; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - $15,000 to $24,999': 'l25000'},
        inplace=True)
    df_census_econ.rename(columns={
        'Estimate; COMMUTING TO WORK - Mean travel time to work (minutes)': 'Mean Travel Time to Work Estimate (minutes)'},
        inplace=True)
    df_census_econ.rename(columns={
        'Percent; INCOME AND BENEFITS (IN 2017| INFLATION-ADJUSTED DOLLARS) - Total households - $200,000 or more': 'Percent of Households with Income Greater than $200,000'},
        inplace=True)
    df_census_econ.rename(columns={
        'Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)': 'Median Household Income Estimate (dollars)'},
        inplace=True)
    df_census_econ.rename(columns={
        'Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Mean household income (dollars)': 'Mean Household Income Estimate (dollars)'},
        inplace=True)
    df_census_econ.rename(columns={
        'Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Per capita income (dollars)': 'Per Capita Income Estimate (dollars)'},
        inplace=True)
    df_census_econ.rename(columns={
        'Percent; HEALTH INSURANCE COVERAGE - Civilian noninstitutionalized population - No health insurance coverage': 'Percent of Population with no Health Insurance Coverage'},
        inplace=True)
    df_census_econ.rename(columns={
        'Percent; PERCENTAGE OF FAMILIES AND PEOPLE WHOSE INCOME IN THE PAST 12 MONTHS IS BELOW THE POVERTY LEVEL - All people': 'Percent of People whose Income in the Past 12 months has been Below Poverty Level'},
        inplace=True)

    df_census_econ['l10000'].replace("-", "0.0", regex=True, inplace=True)
    df_census_econ['l10000'].replace("N", "0.0", regex=True, inplace=True)
    df_census_econ['l10000'] = df_census_econ['l10000'].astype(float)

    df_census_econ['l15000'].replace("-", "0.0", regex=True, inplace=True)
    df_census_econ['l15000'].replace("N", "0.0", regex=True, inplace=True)
    df_census_econ['l15000'] = df_census_econ['l15000'].astype(float)

    df_census_econ['l25000'].replace("-", "0.0", regex=True, inplace=True)
    df_census_econ['l25000'].replace("N", "0.0", regex=True, inplace=True)
    df_census_econ['l25000'] = df_census_econ['l25000'].astype(float)
    df_census_econ["Percent of Households With Income Less Than $24,999"] = df_census_econ['l10000'] + df_census_econ[
        'l15000'] + df_census_econ['l25000']
    df_census_econ = df_census_econ.filter(census_econ_columns)

    df_census_dem.rename(columns={'Id2': 'Zipcode'}, inplace=True)
    df_census_dem.rename(columns={'Estimate; SEX AND AGE - Median age (years)': 'Median Age'}, inplace=True)
    df_census_dem.rename(columns={'Percent; SEX AND AGE - Under 18 years': 'Percent of People under 18 years of age'},
                         inplace=True)
    df_census_dem.rename(columns={'Percent; SEX AND AGE - 65 years and over': 'Percent of People 65 years and over'},
                         inplace=True)
    df_census_dem.rename(columns={'Percent; SEX AND AGE - 18 years and over - Male': 'Percent of Males'}, inplace=True)
    df_census_dem.rename(columns={'Percent; SEX AND AGE - 18 years and over - Female': 'Percent of Females'},
                         inplace=True)
    df_census_dem.rename(columns={
        'Percent; HISPANIC OR LATINO AND RACE - Total population - Hispanic or Latino (of any race)': 'Percent of People who are Hispanic'},
        inplace=True)
    df_census_dem.rename(columns={
        'Percent; HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - White alone': 'Percent of People who are White'},
        inplace=True)
    df_census_dem.rename(columns={
        'Percent; HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - Black or African American alone': 'Percent of People who are Black or African American'},
        inplace=True)
    df_census_dem.rename(columns={
        'Percent; HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - Asian alone': 'Percent of People who are Asian'},
        inplace=True)

    df_census_dem = df_census_dem.filter(census_dem_columns)

    # filter data to only Silicon Valley + San Francisco Zip Codes
    df_census_dem = df_census_dem[df_census_dem['Zipcode'].isin(zip_list)]
    df_census_econ = df_census_econ[df_census_econ['Zipcode'].isin(zip_list)]
    return df_census_econ, df_census_dem


def wrangle_real_estate_headers(df):
    '''
    run before joining dataframes so keys match

    df_sale_counts_by_zip_silicon_valley.columns = df_sale_counts_by_zip_silicon_valley.columns.str.replace('Sales Counts ', '')
    df_sale_counts_by_zip_silicon_valley = df_sale_counts_by_zip_silicon_valley.add_prefix('Sales Counts ')
    df_sale_counts_by_zip_silicon_valley.rename(columns = {'Sales Counts RegionName':'Zipcode'}, inplace=True)

    '''
    df.columns = df.columns.str.replace('All Homes ', '')
    df = df.add_prefix('All Homes ')
    df.rename(columns={'All Homes RegionName': 'Zipcode'}, inplace=True)
    return df


def wrangle_ipo_headers(df):
    df.rename(columns={'Ticker': 'Symbol'}, inplace=True)
    df["Found"] = df["Found"].astype(dtype=np.int64)
    return df


def join_data(df1, df2, key, join_type):
    df = df1.set_index(key).join(df2.set_index(key), how=join_type)
    return df


def merge_data(df1, df2, key):
    df = pd.merge(df1, df2, on=key, how='inner')
    return df


def df_replace(df, replace_list):
    for i in range(0, len(replace_list)):
        df = df.replace([replace_list[i]], [''], regex=True)
    return df


def drop_columns_and_nans(df, drop_columns, nan_columns):
    df = df.drop(['IPO Name', 'Offer date', 'CUSIP', 'PERM'], axis=1)
    for i in range(0, len(nan_columns)):
        df.drop_duplicates(subset=nan_columns[i], keep='first', inplace=True)
    return df


def calculate_distance_between_zips(zipcode, min_radius, max_radius):
    # api-endpoint 
    URL_base = "https://api.zip-codes.com/ZipCodesAPI.svc/1.0/FindZipCodesInRadius?zipcode="

    URL = URL_base + zipcode + '&minimumradius=' + min_radius + '&maximumradius=' + max_radius + '&key=UNBQ2435TAEYA5EIC8J6'
    # sending get request and saving the response as response object 
    contents = urllib.request.urlopen(URL).read()

    # printing the output 
    zipcodes_nearby = []
    print(json.loads(contents))
    for i in range(1, len(json.loads(contents)['DataList'])):
        zipcodes_nearby.append(json.loads(contents)['DataList'][i]['Code'])
    return zipcodes_nearby


def create_zipcode_distances_dictionary(zipcodes, zip_list):
    '''
    ***DONT RUN IF THESE ARE ALREADY CREATED***
    currently stored as data/processed/zipcodes_within_radius.txt
    '''
    print(len(zip_list))
    for i in range(0, len(zip_list)):
        zipcodes[zip_list[i]] = calculate_distance_between_zips(zip_list[i], '0', '5'), calculate_distance_between_zips(
            zip_list[i], '5', '10')
    return zipcodes


def create_text_file_from_dictionary(filename, dictionary):
    '''
    with open('data/processed/zipcodes_within_radius.txt', 'w') as json_file:
          json.dump(zipcodes, json_file)
      '''
    with open(filename, 'w') as json_file:
        json.dump(dictionary, json_file)
    return dictionary


def export_dataframe_to_dictionary(df, name):
    filename = 'data/processed/' + name + '.csv'
    export_csv = df.to_csv(filename, index=True, header=True)  # Don't forget to add '.csv' at the end of the path

def update_zipcodes_dict(zipcodes, zip_list):
    exists = os.path.isfile('../data/processed/zipcodes_within_radius.txt')
    if not exists:
        zipcodes = create_zipcode_distances_dictionary(zipcodes, zip_list)
        create_text_file_from_dictionary('../data/processed/zipcodes_within_radius.txt', zipcodes)
    else:
        zipcodes = {}
        with open('../data/processed/zipcodes_within_radius.txt', 'r') as f:
            zipcodes = json.load(f)
    return zipcodes


def create_IPO_an_Zipcode_dataframe(census_econ_cols, census_dem_cols, df_ipo, df_zip, zipcodes):
    if 'Zipcode' in census_econ_cols:
        census_econ_cols.remove('Zipcode')

    if 'Zipcode' in census_dem_cols:
        census_dem_cols.remove('Zipcode')
    ipo_header_list = list(df_ipo.columns.values) + census_econ_cols + census_dem_cols + ['All Homes Date Filed',
                                                                                          'All Homes Lockup Expiration Date',
                                                                                          'All Homes 1 Year After Date Filed',
                                                                                          'All Homes 2 Years After Date Filed']
    '''
    Distance from IPO   = estimate is .2 if in the same zipcode as IPO
                        = estimate is 0.5 if not in same zip code as IPO and less than 5 miles from zipcode to IPO
                        = estimate is 1 if greater than 5 and less than 10 miles from zipcode to IPO
    '''
    new_df_list = []

    for index, row in df_ipo.iterrows():
        ipo_zipcode = row['Zipcode']
        zipcode_row = df_zip.loc[df_zip['Zipcode'] == int(ipo_zipcode)]
        headerList = join_IPO_and_Zip_Data(row['Date Filed'], row['Lockup Expiration Date'], census_econ_cols,
                                           census_dem_cols)
        data = np.concatenate((np.array(row.values), zipcode_row.filter(headerList).values), axis=None)
        dictionary = dict(zip(ipo_header_list, data))
        dictionary['Symbol'] = index
        dictionary['Distance to IPO'] = .2
        dictionary['Zipcode for Distance'] = ipo_zipcode
        new_df_list.append(dictionary)

        within_5miles = zipcodes[ipo_zipcode][0]
        within_10miles = zipcodes[ipo_zipcode][1]
        for i in range(0, len(within_5miles)):
            zipcode_row = df_zip.loc[df_zip['Zipcode'] == int(within_5miles[i])]
            data = np.concatenate((np.array(row.values), zipcode_row.filter(headerList).values), axis=None)
            dictionary = dict(zip(ipo_header_list, data))
            dictionary['Symbol'] = index
            dictionary['Distance to IPO'] = .5
            dictionary['Zipcode for Distance'] = within_5miles[i]
            new_df_list.append(dictionary)

        for j in range(0, len(within_10miles)):
            zipcode_row = df_zip.loc[df_zip['Zipcode'] == int(within_10miles[j])]
            data = np.concatenate((np.array(row.values), zipcode_row.filter(headerList).values), axis=None)
            dictionary = dict(zip(ipo_header_list, data))
            dictionary['Symbol'] = index
            dictionary['Distance to IPO'] = 1
            dictionary['Zipcode for Distance'] = within_10miles[j]
            new_df_list.append(dictionary)
    ipo_final_df = pd.DataFrame(new_df_list)
    ipo_final_df.dropna(subset=['Median Age'], how='all', inplace=True)
    return ipo_final_df


def normalize_IPO_an_Zipcode_dataframe(normalization_list, df_ipo):
    df_ipo = df_ipo.replace(['--'], [''], regex=True)
    df_ipo = df_ipo.replace(r'^\s*$', np.nan, regex=True)

    df_ipo = df_ipo.replace(['\,'], [''], regex=True)
    df_ipo = df_ipo.replace(['\+'], [''], regex=True)

    scaler = MinMaxScaler()
    df_ipo[normalization_list] = scaler.fit_transform(df_ipo[normalization_list])
    return df_ipo


def join_IPO_and_Zip_Data(IPO_Date_Filed, IPO_Lockup_Expiration_Date, census_econ_cols, census_dem_cols):
    filtered_columns = census_dem_cols +census_econ_cols # remove 'zipcode'
    ipo_month_filed = IPO_Date_Filed.month
    ipo_year_filed = IPO_Date_Filed.year

    AllHomes_header_filed = 'All Homes ' + str(ipo_year_filed) + '-' + str(ipo_month_filed).zfill(2)
    ipo_month = IPO_Lockup_Expiration_Date.month
    ipo_year = IPO_Lockup_Expiration_Date.year
    AllHomes_header_lockup = 'All Homes ' + str(ipo_year) + '-' + str(ipo_month).zfill(2)

    AllHomes_header_filed_1_yr = 'All Homes ' + str(int(ipo_year_filed) + 1) + '-' + str(ipo_month_filed).zfill(2)

    AllHomes_header_filed_2_yr = 'All Homes ' + str(int(ipo_year_filed) + 2) + '-' + str(ipo_month_filed).zfill(2)

    filtered_columns = filtered_columns + [AllHomes_header_filed, AllHomes_header_lockup,
                                           AllHomes_header_filed_1_yr,
                                           AllHomes_header_filed_2_yr]
    return filtered_columns

def update_ipo_list():
    web_scrapers.add_new_ipo_data_to_csv(
        '/Users/aaron/Development/SF-home-price-prediction/data/processed/1997-04_2019_full_ipo_data.csv', 2019, 6, 6)
    df_ipo_list = load_data(['../data/processed/1997-04_2019_full_ipo_data.csv', '../data/raw/ipo_ritter_data.csv'])
    zipcodes, zip_list = create_zipcode_list(
        ['../data/raw/Santa_Clara_County_Zipcodes.csv', '../data/raw/San_Mateo_County_Zipcodes.csv',
         '../data/raw/San_Francisco_County_Zipcodes.csv', '../data/raw/Alameda_County_Zipcodes.csv'])
    df_ipo = wrangle_IPO_data(df_ipo_list[0], zip_list)
    df_ipo_ritter = wrangle_ipo_headers(df_ipo_list[1])
    df_ipo = join_data(df_ipo, df_ipo_ritter, 'Symbol', 'left')
    df_ipo = drop_columns_and_nans(df_ipo, ['IPO Name', 'Offer date', 'CUSIP', 'PERM'], ['CIK'])
    df_ipo.to_csv("../data/processed/df_ipo.csv", index=True)

def main():
    df_real_estate = load_real_estate_data('../data/raw/Zip_MedianListingPrice_AllHomes.csv', 'State', 'CA')
    # data processing to load all IPO Data between 1997 and present data. This data has been scraped using code from src/web_scrapers.py
    df_ipo_list = load_data(['../data/processed/1997-04_2019_full_ipo_data.csv', '../data/raw/ipo_ritter_data.csv'])
    df_census_list = load_data(['../data/raw/zip_census_bureau_economic_characteristics_2017.csv',
                                '../data/raw/zip_census_bureau_age_race_2017.csv'])
    zipcodes, zip_list = create_zipcode_list(
        ['../data/raw/Santa_Clara_County_Zipcodes.csv', '../data/raw/San_Mateo_County_Zipcodes.csv',
         '../data/raw/San_Francisco_County_Zipcodes.csv', '../data/raw/Alameda_County_Zipcodes.csv'])
    df_real_estate = wrangle_real_estate_data(df_real_estate, zip_list,
                                              ['City', 'State', 'Metro', 'CountyName', 'SizeRank'])
    df_ipo = wrangle_IPO_data(df_ipo_list[0], zip_list)
    census_econ_columns = ['Zipcode',
                           'Unemployment Rate',
                           'Mean Travel Time to Work Estimate (minutes)',
                           'Percent of Households with Income Greater than $200,000',
                           'Median Household Income Estimate (dollars)',
                           'Mean Household Income Estimate (dollars)',
                           'Per Capita Income Estimate (dollars)',
                           'Percent of Population with no Health Insurance Coverage',
                           'Percent of People whose Income in the Past 12 months has been Below Poverty Level',
                           'Percent of Households With Income Less Than $24,999']
    census_dem_columns = ['Zipcode',
                          'Median Age',
                          'Percent of People under 18 years of age',
                          'Percent of People 65 years and over',
                          'Percent of Males',
                          'Percent of Females',
                          'Percent of People who are Hispanic',
                          'Percent of People who are White',
                          'Percent of People who are Black or African American',
                          'Percent of People who are Asian']
    df_census_econ, df_census_dem = wrangle_census_data(df_census_list[0], df_census_list[1], zip_list,
                                                        census_econ_columns, census_dem_columns)
    df_real_estate = wrangle_real_estate_headers(df_real_estate)
    df_ipo_ritter = wrangle_ipo_headers(df_ipo_list[1])

    df_census = join_data(df_census_econ, df_census_dem, 'Zipcode', 'inner')
    df_zip = merge_data(df_census, df_real_estate, 'Zipcode')
    df_zip = df_replace(df_zip, ['\+', '\,'])
    df_ipo = join_data(df_ipo, df_ipo_ritter, 'Symbol', 'left')
    df_ipo = drop_columns_and_nans(df_ipo, ['IPO Name', 'Offer date', 'CUSIP', 'PERM'], ['CIK'])
    normalization_list = ['Offer Amount', 'Number of Employees', 'Found', 'Median Age',
                          'Percent of People under 18 years of age',
                          'Percent of People 65 years and over',
                          'Percent of Males',
                          'Percent of Females',
                          'Percent of People who are Hispanic',
                          'Percent of People who are White',
                          'Percent of People who are Black or African American',
                          'Percent of People who are Asian',
                          'Unemployment Rate',
                          'Mean Travel Time to Work Estimate (minutes)',
                          'Percent of Households with Income Greater than $200,000',
                          'Median Household Income Estimate (dollars)',
                          'Mean Household Income Estimate (dollars)',
                          'Per Capita Income Estimate (dollars)',
                          'Percent of Population with no Health Insurance Coverage',
                          'Percent of People whose Income in the Past 12 months has been Below Poverty Level',
                          'Percent of Households With Income Less Than $24,999']
    zipcodes = update_zipcodes_dict(zipcodes, zip_list)
    df_ipo_all = create_IPO_an_Zipcode_dataframe(census_econ_columns, census_dem_columns, df_ipo, df_zip, zipcodes)
    df_ipo_all.to_csv("../data/processed/df_ipo_all.csv", index=False)


if __name__ == "__main__":
    print("we are wrangling data")
    main()

#update_ipo_list()