import os
import pandas as pd
import datetime
import talib as tb
import pandas.io.data as web
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import math

def getX(rawData):
    # get rawdata in form of matrix
    highData = np.array(rawData['High'], dtype=np.dtype(float))
    lowData = np.array(rawData['Low'], dtype=np.dtype(float))
    closeData = np.array(rawData['Close'], dtype=np.dtype(float))
    openData = np.array(rawData['Open'], dtype=np.dtype(float))
    
    # get Adj Close
    adjCloseData = np.array(rawData['Adj Close'], dtype=np.dtype(float))
    
    # get VolumeData
    volumeData = np.array(rawData['Volume'], dtype=np.dtype(float))
    
    # OBV
    obv = tb.OBV(adjCloseData,volumeData)
    #obv = np.array(NormalizeX(obv), dtype=np.dtype(float))   
    
    # get RSI6
    rsi6 = tb.RSI(adjCloseData, 6)
    
    # get RSI12
    rsi12 = tb.RSI(adjCloseData, 12)
    
    # get SMA3
    sma3 = tb.SMA(adjCloseData,3)
    #sma3 = np.array(NormalizeX(sma3), dtype=np.dtype(float))
    
    # get EMA6
    ema6 = tb.EMA(adjCloseData, 6)
    
    # get EMA12
    ema12 = tb.EMA(adjCloseData, 6)
    
    # get ATR14
    atr14 = tb.ATR(highData, lowData, closeData, 14)
    
    # get MFI14
    mfi14 = tb.MFI(highData, lowData, closeData, volumeData, 14)
    
    # get ADX14
    adx14 = tb.ADX(highData, lowData, closeData, 14)
    
    # get ADX20
    adx20 = tb.ADX(highData, lowData, closeData, 20)
    
    # get MOM1
    mom1 = tb.MOM(adjCloseData, 1)
    
    # get MOM3
    mom3 = tb.MOM(adjCloseData, 3)
    
    # get CCI12
    cci12 = tb.CCI(highData, lowData, closeData, 12)
    
    # get CCI20
    cci20 = tb.CCI(highData, lowData, closeData, 20)
    
    # get ROCR3
    rocr3 = tb.ROCR100(adjCloseData, 3)
    
    # get ROCR12
    rocr12 = tb.ROCR100(adjCloseData, 12)
    
    # get outMACD, outMACDSignal, outMACDHist
    outMACD, outMACDSignal, outMACDHist = tb.MACD(adjCloseData,fastperiod=12,slowperiod=26,signalperiod=9)
    
    # get WILLR
    willR = tb.WILLR(highData,lowData,closeData)
    #willR = np.array(NormalizeX(willr), dtype=np.dtype(float))
        
    # get TSF10
    tsf10 = tb.TSF(adjCloseData, 10)
    
    # get TSF20
    tsf20 = tb.TSF(adjCloseData, 20)
    
    # get TRIX, default to 12 days
    trix = tb.TRIX(adjCloseData, 12)
    
    # get BBANDSUPPER, BBANDSMIDDLE, BBANDSLOWER
    bupper, bmiddle, blower = tb.BBANDS(adjCloseData, 10, 2, 2, 0)
    
    #Return Train matrix
    return np.column_stack((adjCloseData, obv, volumeData, rsi6, rsi12, sma3, ema6, ema12, atr14, mfi14, adx14, adx20, mom1, mom3, cci12, cci20, rocr3, rocr12, outMACD, outMACDSignal, willR, tsf10, tsf20, trix, bupper, bmiddle, blower))

#Implement Logistic Regression
def Logistic_reg(X,y):
#==============================================================================
#     model = LogisticRegression().fit(X, y)
#     print model.score(X, y)
#==============================================================================
    # evaluate the model by splitting into train and test sets    
    X_train = X[:100,:]
    X_test = X[101:,:]
    y_train = y[:100]
    y_test = y[101:]
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)
    # predict class labels for the training set
    predicted_train = model2.predict(X_train)
    # predict class labels for the test set
    predicted_test = model2.predict(X_test)
    #print predicted
    # generate class probabilities
    #probs = model2.predict_proba(X_test)
    #print probs
    # generate evaluation metrics
    print "Accuracy for training set using Logistic regression"
    print metrics.accuracy_score(y_train, predicted_train)
    print "Accuracy for test set using Logistic regression"
    print metrics.accuracy_score(y_test, predicted_test)
    #print metrics.roc_auc_score(y_test, probs[:, 1])   
    # evaluate the model using 10-fold cross-validation
#==============================================================================
#     scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=5)
#     print scores
#     print scores.mean()
#==============================================================================
    #validation for model
    return model2,y_test
    
    
if __name__ == "__main__": 
    stocks = ['AAPL']
    start = datetime.date(2015,11,1)
    end = datetime.date(2016,6,30)
    rawData = web.DataReader(stocks, 'yahoo',start, end)
    rawData = rawData.to_frame()

    #get X matrix
    X = getX(rawData)
    
    #Normalize X matrix
    for n in range(0,X.shape[1]):
        X[:,n]=X[:,n]/np.nanmax(abs(X[:,n]))
    
    #cut X matrix to remove NaN's
    X_new = X[39:,:]
        
    #Build Y matrix
    sma3 = X_new[:,5]
    Y = [1 if (sma3[x] - sma3[x+3])<0 else -1 for x in range(0,len(sma3)-3)]
    #Y=Y+[1,1,1]
    
    X_new = X_new[:125,:]    
    
    #get Logistic regression    
    model2,y_test = Logistic_reg(X_new,Y)
    
