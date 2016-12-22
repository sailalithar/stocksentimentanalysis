# -*- coding: utf-8 -*-
"""
@author: Shruti
@author: SaiLalitha
@author Nishit
"""

import os, pandas as pd, pandas.io.data as web, datetime, talib as tb, matplotlib.pyplot as plt, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from LR import LogisticRegressionImp
from stock_list import StockList
from SVM import SupportVectorMachine


if __name__ == "__main__":
    
     #Start and end Time
    start = datetime.date(2015,11,1)
    end = datetime.date(2016,7,15)
    
    # creates a list of all stocks in 8 category
    stockList = StockList()
    stocks = stockList.createList()
    
    #call LR
    lr = LogisticRegressionImp()
    
    print 'Evaluating Logisitc Regression'
    df, feature = lr.evaluate(stocks,start, end)
    df = pd.DataFrame(df,columns = ['Stock Category','Stock Name','LR 27 train', 'LR 27 test', 'LR 54 train', 'LR 54 test', 'LR 8 train', 'LR 8 test', 'LR 16 train', 'LR 16 test'] )
    df.to_excel('LR.xlsx', sheet_name='sheet1', index=False)
    feature = pd.DataFrame(feature,columns = ['Stock Category','Stock Name','Selected Features', 'Train Accuracy', 'Test Accuracy'] )
    feature.to_excel('Select Feature Accuracy LR.xlsx', sheet_name='sheet1', index=False)
        
    
    #call svm
    svm = SupportVectorMachine()
    print 'Evaluating Support Vector Machine'
    df, feature = svm.evaluate(stocks,start, end)
    df = pd.DataFrame(df,columns = ['Stock Category','Stock Name','SVM 27 train', 'SVM 27 test', 'SVM 54 train', 'SVM 54 test', 'SVM 8 train', 'SVM 8 test', 'SVM 16 train', 'SVM 16 test'])
    df.to_excel('SVM.xlsx', sheet_name='sheet1', index=False)
    feature = pd.DataFrame(feature,columns = ['Stock Category','Stock Name','Selected Features', 'Train Accuracy', 'Test Accuracy'] )
    feature.to_excel('Select Feature Accuracy SVM.xlsx', sheet_name='sheet1', index=False)
        
    print 'done'
    
    
