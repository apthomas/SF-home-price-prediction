from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_ipo_data_from_nasdaq():
	'''
	website example format: https://www.nasdaq.com/markets/ipos/activity.aspx?tab=filings&month=1996-09
	initializing variables and dataframes
	'''
	base_url = "https://www.nasdaq.com/markets/ipos/activity.aspx?tab=filings&month="
	base_year = 1997
	end_year = 2019
	base_month = 1
	end_month = 12
	data=[]
	company_data=[]
	headers = ['Company Name', 'Symbol', 'Offer Amount', 'Date Filed']
	company_headers = ['Zipcode, Number of Employees', 'Lockup Period', 'Lockup Expiration Date', 'CIK']
	all_headers = headers+company_headers
	joined_data=[]
	filename = 'test.csv'
	df = pd.DataFrame(data, columns=headers)


	for year in range(base_year, end_year+1):
		for month in range(base_month, end_month+1):
			data=[]
			company_data={}
			page_url = base_url+str(year)+"-"+str(month)
			page = urlopen(page_url)
			soup = BeautifulSoup(page, 'html.parser')
			table = soup.find('div', attrs={'class':'genTable'})
			table_body = table.find('tbody')
			
			rows = table_body.find_all('tr')
			for row in rows:
			    cols = row.find_all('td')
			    cols = [ele.text.strip() for ele in cols]
			    cols[2] = cols[2].replace('$','')
			    cols[2] = cols[2].replace(',','')
			    if (len(cols) == len(headers)):
			    	data.append([ele for ele in cols if ele]) # Get rid of empty values
			if (len(data[0])== len(headers)):
				links = create_list_of_href(table_body)
				for link in links:
					c_data, symbol = grab_data_from_company_ipo(link)
					company_data[symbol] = c_data
				for row in data:
					if row[1] in company_data:
						joined_data.append(row+company_data[row[1]])
				df2 = pd.DataFrame(joined_data, columns=all_headers)
				df2.dropna(inplace=True)
				df = pd.concat([df, df2], ignore_index=True)
			print(str(month)+" month completed")

		print(str(year)+" completed")

		
	print(df.describe())
	df.to_csv(filename, encoding='utf-8', index=False)

def create_list_of_href(table_body):
	links = table_body.find_all(href=True)
	del links[1::2] # remove all odd links
	for i in range(0, len(links)):
		quote_list = [m.start() for m in re.finditer('"',str(links[i]))]
		links[i] = str(links[i])[quote_list[0]+1:quote_list[1]]

	return links

def grab_data_from_company_ipo(link):
	'''
	Example: 'https://www.nasdaq.com/markets/ipos/company/cerus-corp-11233-8004'
	'''
	link_url = link
	page_company = urlopen(link_url)
	soup_company = BeautifulSoup(page_company, 'html.parser')
	table_company = soup_company.find('div', attrs={'class':'genTable'})
	table_body_company = table_company.find('tbody')
	zipcode = soup_company.find("td", text="Company Address").find_next_sibling("td").text
	zipcode = determine_zip_code(zipcode)
	employees = soup_company.find("td", text=re.compile("Employees")).find_next_sibling("td").text
	symbol = soup_company.find("td", text="Proposed Symbol").find_next_sibling("td").text
	lockupPeriod = soup_company.find("td", text=re.compile("Lockup Period")).find_next_sibling("td").text
	lockupExpiration = soup_company.find("td", text="Lockup Expiration").find_next_sibling("td").text
	cik = soup_company.find("td", text="CIK").find_next_sibling("td").text
	print([zipcode, employees, lockupPeriod, lockupExpiration, cik, symbol])
	return ([zipcode, employees, lockupPeriod, lockupExpiration, cik], symbol)

def determine_zip_code(address):
	if (address[len(address)-5] == '-'):
		return address[len(address)-10:]
	return address[len(address)-5:]

scrape_ipo_data_from_nasdaq()
#track_href_of_ticker()
#grab_data_from_company_ipo('https://www.nasdaq.com/markets/ipos/company/cerus-corp-11233-8004')