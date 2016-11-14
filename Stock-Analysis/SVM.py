# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:33:43 2016

@author: Shruti Kamtekar
"""

from GenerateData import GenerateData
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas.io.data as web
from sklearn.linear_model import LogisticRegression



def Logistic_reg(train_X,test_X,train_y):
    clf2 = LogisticRegression().fit(train_X, train_y)
    LogisticRegression(C=1.0, dual=False, fit_intercept=True, intercept_scaling=1,
          penalty='l2', tol=0.0001)
    print clf2.predict_proba(test_X)

if __name__ == '__main__':
    dataset = GenerateData()
    stocks = ['AAPL']
    start = datetime.date(2016,1,1)
    end = datetime.date(2016,7,31)
	
    rawData = web.DataReader(stocks, 'yahoo',start, end)
    rawData = rawData.to_frame()
    X, y = dataset.GetData(rawData)

    index = int(0.6*len(X))
    train_X = X[:index,:]
    test_X = X[index:,:]
    train_y = y[:index]
    test_y = y[index:]
    Logistic_reg(train_X,test_X,train_y)
