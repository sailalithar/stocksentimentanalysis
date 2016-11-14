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
from sklearn import svm




def Logistic_reg(X_train, y_train, X_test, y_test):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    LogisticRegression(C=1.0, dual=False, fit_intercept=True, intercept_scaling=1,
          penalty='l2', tol=0.0001)
    print 'accuracy for logsitic regression' 
    print clf.score(X_test, y_test)
     
def RBF_kernel(X_train, y_train, X_test, y_test):     
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train) 
    print 'accuracy for rbf svm' 
    print clf.score(X_test, y_test)
    
if __name__ == '__main__':
    dataset = GenerateData()
    stocks = ['AAPL']
    start = datetime.date(2000,1,1)
    end = datetime.date(2016,10,31)
	
    rawData = web.DataReader(stocks, 'yahoo',start, end)
    rawData = rawData.to_frame()
    X, y = dataset.GetData(rawData)

#    stocks = ['^IXIC']
#    rawData = web.DataReader(stocks, 'yahoo',start, end)
#    rawData = rawData.to_frame()
#    X_nasdaq, y_nasdaq = dataset.GetData(rawData)
#    
#    X = np.column_stack((X_appl,X_nasdaq))
#    y = np.concatenate((y_appl,y_nasdaq))
    
    index = int(0.6*len(X))
    X_train = X[:index,:]
    X_test = X[index:,:]
    y_train = y[:index]
    y_test = y[index:]
#    
    Logistic_reg(X_train, y_train, X_test, y_test)
    RBF_kernel(X_train, y_train, X_test, y_test)
    
    