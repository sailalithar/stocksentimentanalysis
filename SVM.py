# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:33:43 2016

@author: Shruti Kamtekar
"""

import numpy as np
from sklearn import svm
import pandas as pd
import datetime
import pandas.io.data as web
import talib as tb


stocks = ['AAPL']
start = datetime.date(2016,1,1)
end = datetime.date(2016,3,31)
rawData = web.DataReader(stocks, 'yahoo',start, end)
rawData = rawData.to_frame()

# get rawdata in form of matrix
highData = np.array(rawData['High'], dtype=np.dtype(float))
lowData = np.array(rawData['Low'], dtype=np.dtype(float))
closeData = np.array(rawData['Close'], dtype=np.dtype(float))
openData = np.array(rawData['Open'], dtype=np.dtype(float))
volumeData = np.array(rawData['Volume'], dtype=np.dtype(float))

# get features
willr = tb.WILLR(highData,lowData,closeData)
obv = tb
##print features
