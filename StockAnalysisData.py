# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:56:29 2016

@author: Shruti Kamtekar
"""

import os
import pandas as pd
import datetime
import pandas.io.data as web


def CleanData(rawData):
    #Calculate Adj Close feature
    adjCloseData = rawData.ix['Adj Close']
    adjCloseData.columns = ['Adj Close']
    print adjCloseData
    #Calculate WILLR feature
    willRData = (rawData.ix['High']-rawData.ix['Close'])*100/(rawData.ix['High']-rawData.ix['Low'])
    willRData.columns =['WILLR']
    print willRData
    #calculate OBV
    for column in rawData.ix['Volume']:
        
    
    
    
#    dataFrame = dataFrame.append(pd.DataFrame(willRData,columns =['WILLR']))
#    print dataFrame
    
stocks = ['AAPL']
start = datetime.date(2000,1,1)
end = datetime.date(2016,11,11)
f = web.DataReader(stocks, 'yahoo',start, end)
CleanData(f)


