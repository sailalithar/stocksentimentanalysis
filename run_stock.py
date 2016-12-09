# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 09:37:37 2016

@author: Shruti
"""

import os, pandas as pd, pandas.io.data as web, datetime, talib as tb, matplotlib.pyplot as plt, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from LR import LogisticRegressionImp
from stock_list import StockList


if __name__ == "__main__":
    
     #Start and end Time
    start = datetime.date(2015,11,1)
    end = datetime.date(2016,7,15)
    
    # creates a list of all stocks in 8 category
    stockList = StockList()
    stocks = stockList.createList()
    
    #call LR
    lr = LogisticRegressionImp()
    
    all_stock, stock_SnP, all_stock_ER, stock_SnP_ER = lr.evaluate(stocks,start, end)
    print 'done'
    
    
