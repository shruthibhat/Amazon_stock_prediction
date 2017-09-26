
# coding: utf-8

# In[1]:

'''
Author:Shruthi Bhat

INTRODUCTION:

This is a Web scraping project which crawls into Yahoo finance website and 
programmatically collects one month historical data of AMZN stock using 
Python modules beautifulsoup and requests and writes to a CSV file.Based on
the data collected,opening stock price for current day is predicted using 
linear regression.Python module scikit learn is used to program linear 
regression and Python modules matplotlib is used to plot stock prices and 
the corresponding linear regression line in the graph,stock's High values 
and Low values over three months period.
url:https://finance.yahoo.com/quote/AMZN/history?period1=1495090800&period2=1503039600&interval=1d&filter=history&frequency=1d)

'''


# In[2]:

'''

REQUIREMENTS:

Python version used 2.7

Modules needed:

1.Requests
2.BeautifulSoup
3.Pandas
4.Numpy
5.Matplotlib
6.scikit-learn

'''


# In[3]:

'''
This makes the Jupyter cells wider

'''
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[5]:

'''
The code below uses BeautifulSoup library to scrape July month Amazon stock data from
Yahoo finance website and writes to AMZN_Stock.csv file

'''
import datetime
import requests
import csv
from bs4 import BeautifulSoup

html_doc=requests.get('https://finance.yahoo.com/quote/AMZN/history?period1=1498892400&period2=1501484400&interval=1d&filter=history&frequency=1d')
soup=BeautifulSoup(html_doc.content,'html.parser')
div=soup.find(id="Col1-1-HistoricalDataTable-Proxy") #Col1-1-HistoricalDataTable-Proxy #Col1-1-QuoteLeaf-Proxy
table=div.select_one("table")
#tbody=table.select_one("tbody")

headers = [th.text.encode("utf-8") for th in table.select("tr th")]

with open("AMZN_Stock.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows([[td.text.encode("utf-8") for td in row.find_all("td")] for row in table.select("tr + tr")])


# In[6]:

'''
Pandas module is used to read the values from csv file and display it
'''
import pandas as pd
import datetime
df = pd.read_csv('AMZN_Stock.csv',names=['Date','Open','High','Low','Close','Adj Close','Volume'],parse_dates=['Date'],thousands=',')
df


# In[7]:

'''
The graph below shows the stock's High value for each day in one month period

'''

import numpy as np  
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

high_list=list(df['High'])
date_list=list(df['Date'])
plt.plot(date_list,high_list)
plt.xticks(rotation=30)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[8]:

'''
The graph below shows the stock's Low value for each day in one month period

'''

import numpy as np  
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

low_list=list(df['Low'])
date_list=list(df['Date'])
plt.plot(date_list,low_list)
plt.xticks(rotation=30)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[9]:

'''
matplotlib and scikit learn module are used to plot graph and 
calculate simple linear regression respectively

'''
import datetime
import time
import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

 
dates = []
prices = []
 
def get_data(filename):
    '''
        This method is used to read the values from the columns corresponding to date and
        opening price and append the dates and prices list respectively
    '''    
    with open(filename,'Ur') as csvfile:
        csvFileReader = csv.reader(csvfile)
        for row in csvFileReader:
            dates.append(int(row[0].split(",")[0].split(" ")[1]))
            prices.append(float(row[4].replace(',','')))
    return
 
def show_plot(dates,prices):
    '''
        This method is used to calculate the simple linear regression
    '''    
    linear_mod = linear_model.LinearRegression()
    dates = np.reshape(dates,(len(dates),1)) # converting to matrix of n X 1
    prices = np.reshape(prices,(len(prices),1))
    linear_mod.fit(dates,prices) #fitting the data points in the model
    plt.scatter(dates,prices,color='black') #plotting the initial datapoints 
    plt.plot(dates,linear_mod.predict(dates),color='red',linewidth=3) #plotting the line made by linear regression
    plt.show()
    return
 
def predict_price(dates,prices,x):
    linear_mod = linear_model.LinearRegression() #defining the linear regression model
    dates = np.reshape(dates,(len(dates),1)) # converting to matrix of n X 1
    prices = np.reshape(prices,(len(prices),1))
    linear_mod.fit(dates,prices) #fitting the data points in the model
    predicted_price =linear_mod.predict(x)
    return predicted_price[0][0],linear_mod.coef_[0][0] ,linear_mod.intercept_[0]
 
get_data('AMZN_Stock.csv') # calling get_data method by passing the csv file to it
print dates
print prices
print "\n"
 
show_plot(dates,prices) 
 


today=datetime.datetime.now()
#passing today's date to the model
predicted_price, coefficient, constant = predict_price(dates,prices,today.day)  
print "CONCLUSION:"
print "Based on July's Data, predicted price for AMZN stock for today({}) is:   ${} ".format(time.strftime("%m/%d/%Y"),str(predicted_price))
print "The regression coefficient is ",str(coefficient),", and the constant is ", str(constant)
print "the relationship equation between dates and prices is: price = ",str(coefficient),"* date + ",str(constant)




# In[ ]:

'''
References:
1.https://www.analyticsvidhya.com/blog/2015/10/beginner-guide-web-scraping-beautiful-soup-python/
2.https://medium.freecodecamp.org/how-to-scrape-websites-with-python-and-beautifulsoup-5946935d93fe
3.https://www.dataquest.io/blog/web-scraping-tutorial-python/
4.https://automatetheboringstuff.com/chapter11/
5.https://www.cyberciti.biz/faq/howto-get-current-date-time-in-python/
6.http://beancoder.com/linear-regression-stock-prediction/
7.https://medium.com/towards-data-science/simple-and-multiple-linear-regression-in-python-c928425168f9
'''


