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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
import math

def getX(rawData, rawDataSnP):
    
    # get rawdata in form of matrix
    highData = np.array(rawData['High'], dtype=np.dtype(float))
    lowData = np.array(rawData['Low'], dtype=np.dtype(float))
    closeData = np.array(rawData['Close'], dtype=np.dtype(float))
    openData = np.array(rawData['Open'], dtype=np.dtype(float))


    #SnP
    highDataSnP = np.array(rawDataSnP['High'], dtype=np.dtype(float))
    lowDataSnP = np.array(rawDataSnP['Low'], dtype=np.dtype(float))
    closeDataSnP = np.array(rawDataSnP['Close'], dtype=np.dtype(float))
    openDataSnP = np.array(rawDataSnP['Open'], dtype=np.dtype(float))    
    
    # get Adj Close
    adjCloseData = np.array(rawData['Adj Close'], dtype=np.dtype(float))
    adjCloseDataSnP = np.array(rawDataSnP['Adj Close'], dtype=np.dtype(float))
    
    # get VolumeData
    volumeData = np.array(rawData['Volume'], dtype=np.dtype(float))
    volumeDataSnP = np.array(rawDataSnP['Volume'], dtype=np.dtype(float))
    
    # OBV
    obv = tb.OBV(adjCloseData,volumeData)
    obvSnP = tb.OBV(adjCloseDataSnP,volumeDataSnP)
    #obv = np.array(NormalizeX(obv), dtype=np.dtype(float))   
    
    # get RSI6
    rsi6 = tb.RSI(adjCloseData, 6)
    rsi6SnP = tb.RSI(adjCloseDataSnP, 6)    
    
    # get RSI12
    rsi12 = tb.RSI(adjCloseData, 12)
    rsi12SnP = tb.RSI(adjCloseDataSnP, 12)
    
    # get SMA3
    sma3 = tb.SMA(adjCloseData, 3)
    sma3SnP = tb.SMA(adjCloseDataSnP, 3)
    #sma3 = np.array(NormalizeX(sma3), dtype=np.dtype(float))
    
    # get EMA6
    ema6 = tb.EMA(adjCloseData, 6)
    ema6SnP = tb.EMA(adjCloseDataSnP, 6)
    
    # get EMA12
    ema12 = tb.EMA(adjCloseData, 6)
    ema12SnP = tb.EMA(adjCloseDataSnP, 6)
    
    # get ATR14
    atr14 = tb.ATR(highData, lowData, closeData, 14)
    atr14SnP = tb.ATR(highDataSnP, lowDataSnP, closeDataSnP, 14)
    
    # get MFI14
    mfi14 = tb.MFI(highData, lowData, closeData, volumeData, 14)
    mfi14SnP = tb.MFI(highDataSnP, lowDataSnP, closeDataSnP, volumeDataSnP, 14)
    
    # get ADX14
    adx14 = tb.ADX(highData, lowData, closeData, 14)
    adx14SnP = tb.ADX(highDataSnP, lowDataSnP, closeDataSnP, 14)
    
    # get ADX20
    adx20 = tb.ADX(highData, lowData, closeData, 20)
    adx20SnP = tb.ADX(highDataSnP, lowDataSnP, closeDataSnP, 20)
    
    # get MOM1
    mom1 = tb.MOM(adjCloseData, 1)
    mom1SnP = tb.MOM(adjCloseDataSnP, 1)
    
    # get MOM3
    mom3 = tb.MOM(adjCloseData, 3)
    mom3SnP = tb.MOM(adjCloseDataSnP, 3)
    
    # get CCI12
    cci12 = tb.CCI(highData, lowData, closeData, 12)
    cci12SnP = tb.CCI(highDataSnP, lowDataSnP, closeDataSnP, 12)
    
    # get CCI20
    cci20 = tb.CCI(highData, lowData, closeData, 20)
    cci20SnP = tb.CCI(highDataSnP, lowDataSnP, closeDataSnP, 20)
    
    # get ROCR3
    rocr3 = tb.ROCR100(adjCloseData, 3)
    rocr3SnP = tb.ROCR100(adjCloseDataSnP, 3)
    
    # get ROCR12
    rocr12 = tb.ROCR100(adjCloseData, 12)
    rocr12SnP = tb.ROCR100(adjCloseDataSnP, 12)
    
    # get outMACD, outMACDSignal, outMACDHist
    outMACD, outMACDSignal, outMACDHist = tb.MACD(adjCloseData,fastperiod=12,slowperiod=26,signalperiod=9)
    outMACDSnP, outMACDSignalSnP, outMACDHistSnP = tb.MACD(adjCloseDataSnP,fastperiod=12,slowperiod=26,signalperiod=9)
    
    # get WILLR
    willR = tb.WILLR(highData,lowData,closeData)
    willRSnP = tb.WILLR(highDataSnP,lowDataSnP,closeDataSnP)
    #willR = np.array(NormalizeX(willr), dtype=np.dtype(float))
        
    # get TSF10
    tsf10 = tb.TSF(adjCloseData, 10)
    tsf10SnP = tb.TSF(adjCloseDataSnP, 10)
    
    # get TSF20
    tsf20 = tb.TSF(adjCloseData, 20)
    tsf20SnP = tb.TSF(adjCloseDataSnP, 20)
    
    # get TRIX, default to 12 days
    trix = tb.TRIX(adjCloseData, 12)
    trixSnP = tb.TRIX(adjCloseDataSnP, 12)
    
    # get BBANDSUPPER, BBANDSMIDDLE, BBANDSLOWER
    bupper, bmiddle, blower = tb.BBANDS(adjCloseData, 10, 2, 2, 0)
    bupperSnP, bmiddleSnP, blowerSnP = tb.BBANDS(adjCloseDataSnP, 10, 2, 2, 0)
    
    df = pd.DataFrame(
                        {
                        'adjCloseData' : adjCloseData, 'obv' : obv,
                        'volumeData' : volumeData, 'rsi6' : rsi6, 'rsi12' : rsi12,
                        'sma3' : sma3, 'ema6' : ema6, 'ema12' : ema12,
                        'atr14' : atr14, 'mfi14' : mfi14, 'adx14' : adx14,
                        'adx20' : adx20, 'mom1' : mom1, 'mom3' : mom3,
                        'cci12' : cci12, 'cci20' : cci20, 'rocr3' : rocr3,
                        'rocr12' : rocr12, 'outMACD' : outMACD,
                        'outMACDSignal' : outMACDSignal, 'willR' : willR,
                        'tsf10' : tsf10, 'tsf20' : tsf20, 'trix' : trix,
                        'bupper' : bupper, 'bmiddle': bmiddle, 'blower' : blower,
                        
                        'adjCloseDataSnP' : adjCloseDataSnP, 'obvSnP' : obvSnP,
                       'volumeDataSnP' : volumeDataSnP, 'rsi6SnP' : rsi6SnP,
                       'rsi12SnP' : rsi12SnP, 'sma3SnP' : sma3SnP,
                       'ema6SnP' : ema6SnP, 'ema12SnP' : ema12SnP,
                       'atr14SnP' : atr14SnP, 'mfi14SnP' : mfi14SnP,
                       'adx14SnP' : adx14SnP, 'adx20SnP' : adx20SnP,
                       'mom1SnP' : mom1SnP, 'mom3SnP' : mom3SnP,
                       'cci12SnP' : cci12SnP, 'cci20SnP' : cci20SnP,
                       'rocr3SnP' : rocr3SnP, 'rocr12SnP' : rocr12SnP,
                       'outMACDSnP' : outMACDSnP,
                       'outMACDSignalSnP' : outMACDSignalSnP, 'willRSnP' : willRSnP,
                       'tsf10SnP' : tsf10SnP, 'tsf20SnP' : tsf20SnP,
                       'trixSnP' : trixSnP, 'bupperSnP' : bupperSnP,
                       'bmiddleSnP': bmiddleSnP, 'blowerSnP' : blowerSnP
                       }
                    )   
   
    #Return Training dataframe
    return df

#Implement Logistic Regression
def Logistic_reg(X,y):
#==============================================================================
#     model = LogisticRegression().fit(X, y)
#     print model.score(X, y)
#==============================================================================
    # evaluate the model by splitting into train and test sets 
    train = int(X.shape[0] * 0.8)
    test = train + 1
    X_train = X[:train,:]
    X_test = X[test:,:]
    y_train = y[:train]
    y_test = y[test:]
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)
    
    # predict class labels for the training set
    predicted_train = model2.predict(X_train)

    # predict class labels for the test set
    predicted_test = model2.predict(X_test)
    
    # generate evaluation metrics
    print "Accuracy for training set using Logistic regression"
    acc_train = metrics.accuracy_score(y_train, predicted_train)
    print acc_train

    print "Accuracy for test set using Logistic regression"
    acc_test = metrics.accuracy_score(y_test, predicted_test)
    print acc_test

    #validation for model
    return model2, y_test, acc_train, acc_test
    
def feat_select(X,y):
    
    #feature selection using extermely randomized tree algorithm 
    model = ExtraTreesClassifier()
    model.fit(X, y)    

    return model.feature_importances_    

def evaluate(stocks, start, end):
    
    for stock in stocks:
        
        rawData = web.DataReader([stock], 'yahoo',start, end)

        rawData = rawData.to_frame()
        
        #S&P
        rawDataSnP = (web.DataReader(['^GSPC'], 'yahoo',start, end)).to_frame()
    
        #get X matrix
        df = getX(rawData, rawDataSnP)
        
        #convert dataframe to training matrix
        X = pd.DataFrame.as_matrix(df)
    
        #Adding SnP and Nasdaq's features to X matrix
        #dfSnP = getX(rawDataSnP)
#        df2 = getX(rawData2)
    
        #Normalize X matrix
        for n in range(0,X.shape[1]):
            X[:,n]=X[:,n]/np.nanmax(abs(X[:,n]))
        
        #cut X matrix to remove NaN's
        X_new = X[39:,:]
            
        #Build Y matrix
        sma3 = X_new[:,5]
        Y = [1 if (sma3[x] - sma3[x+3])<0 else -1 for x in range(0,len(sma3)-3)]
        #Y=Y+[1,1,1]
        sma3SnP = X_new[:,32]
        YSnP = [1 if (sma3SnP[x] - sma3SnP[x+3])<0 else -1 for x in range(0,len(sma3SnP)-3)]
        
        X_new = X_new[:-3,:]    
        Y = Y + YSnP
        
        #scores = feat_select(X_new,Y)    
        #get Logistic regression    
        model2,y_test, train_acc, test_acc = Logistic_reg(X_new,Y)
    
        finalList.append(
                        {
                            'stock': stock,
                            'model': model2,
                            'train_acc': train_acc,
                            'test_acc': test_acc
                        }
                    )

        return finalList        
    
if __name__ == "__main__":
    
    #Stock List
    stocks = ['FB']
    
    #Start and end Time
    start = datetime.date(2015,11,1)
    end = datetime.date(2016,7,15)
    
#    rawData2 = (web.DataReader(['^IXIC'], 'yahoo',start, end)).to_frame()    

    finalList = evaluate(stocks, start, end)
