# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:56:29 2016

@author: Shruti Kamtekar
"""

import math
import pandas as pd
import datetime
import pandas.io.data as web
pd.options.mode.chained_assignment = None  # default='warn'



def CalculateOBV(rawData):
    rawData['Diff'] = rawData['Close'].diff().to_frame()
    rawData['UpDown'] = rawData.Diff.map( lambda x:1 if math.isnan(x) or x>=0 else -1)
    obvData = rawData.UpDown*rawData.Volume.cumsum()
    obvData = obvData.to_frame()
    obvData.columns = ['OBV']
    return obvData

def CalculateWillR(rawData):    
    willR = (rawData['High'] - rawData['Close'])*100/(rawData['High'] - rawData['Low'])
    willR = willR.to_frame()
    willR.columns = ['WILLR']
    return willR

def CalculateROCR(rawData):    
    rocR3 =	rawData.div(rawData.shift(3))
    rocR3.columns = ['ROCR3']
    rocR12 = rawData.div(rawData.shift(12))
    rocR12.columns = ['ROCR12']
    print rocR12.head(5)
    return rocR3,rocR12
    
def CalculateMOM(rawData):    
    mom1 = rawData.diff(1)
    mom1.columns = ['MOM1']
    mom3 = rawData.diff(3)
    mom3.columns = ['MOM3']
    return mom1,mom3     
      
      
stocks = ['AAPL','^GSPC','^IXIC']
start = datetime.date(2016,1,1)
end = datetime.date(2016,1,31)
f = web.DataReader(stocks, 'yahoo',start, end)
#f.rename({0: 'blaaa', 2: 5})
print 'Raw extracted Data'
f = f.to_frame()
print f.head(1)
print 
print

# creating feature dataframe from close and volume
features = f[['Adj Close','Volume']]

# creating feature dataframe for obv
obv = CalculateOBV(f[['Close','Volume']])
features = features.join(obv)

# creating feature dataframe for willr
willr = CalculateWillR(f[['High','Close','Low']])
features = features.join(willr)

# creating feature dataframe for rocr
rocR3,rocR12 = CalculateROCR(f[['Adj Close']])
features = features.join(rocR3)
features = features.join(rocR12)

# creating feature dataframe for mom
mom1,mom3 = CalculateMOM(f[['Adj Close']])
features = features.join(mom1)
features = features.join(mom3)

print 'Feature DataFrame'
print features.head(20)
#
##
