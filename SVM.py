import os
import pandas as pd
import datetime
import talib as tb
import pandas.io.data as web
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

def getTrainX(rawData):
    # get rawdata in form of matrix
    highData = np.array(rawData['High'], dtype=np.dtype(float))
    lowData = np.array(rawData['Low'], dtype=np.dtype(float))
    closeData = np.array(rawData['Close'], dtype=np.dtype(float))
    openData = np.array(rawData['Open'], dtype=np.dtype(float))
    
    # get Adj Close
    adjCloseData = np.array(rawData['Adj Close'], dtype=np.dtype(float))
    
    # OBV
    OBV = 
    
    # get VolumeData
    volumeData = np.array(rawData['Volume'], dtype=np.dtype(float))
    
    # get RSI6
    rsi6 = tb.RSI(closeData, 6)
    
    # get RSI12
    rsi12 = tb.RSI(closeData, 12)
    
    # get SMA3
    sma3 = 
    
    # get EMA6
    ema6 = tb.EMA(closeData, 6)
    
    # get EMA12
    ema12 = tb.EMA(closeData, 6)
    
    # get ATR14
    atr14 = tb.ATR(highData, lowData, closeData, 14)
    
    # get MFI14
    mfi14 = tb.MFI(highData, lowData, closeData, volumeData, 14)
    
    # get ADX14
    adx14 = tb.ADX(highData, lowData, closeData, 14)
    
    # get ADX20
    adx20 = tb.ADX(highData, lowData, closeData, 20)
    
    # get MOM1
    mom1 = tb.MOM(closeData, 1)
    
    # get MOM3
    mom3 = tb.MOM(closeData, 3)
    
    # get CCI12
    cci12 = tb.CCI(highData, lowData, closeData, 12)
    
    # get CCI20
    cci20 = tb.CCI(highData, lowData, closeData, 20)
    
    # get ROCR3
    rocr3 = tb.ROCR100(closeData, 3)
    
    # get ROCR12
    rocr12 = tb.ROCR100(closeData, 12)
    
    # get outMACD
    outMACD = 
    
    # get outMACDSignal
    outMACDSignal = 
    
    #get outMACDHist
    outMACDHist = 
    
    # get WILLR
    willr = tb.WILLR(highData,lowData,closeData)
    
    # get TSF10
    tsf10 = tb.TSF(closeData, 10)
    
    # get TSF20
    tsf20 = tb.TSF(closeData, 20)
    
    # get TRIX, default to 12 days
    trix = tb.TRIX(closeData, 12)
    
    # get BBANDSUPPER, BBANDSMIDDLE, BBANDSLOWER
    bupper, bmiddle, blower = tb.BBANDS(closeData, 10, 2, 2, 0)
    
    #Return Train matrix
    return np.column_stack((adjCloseData, OBV, volumeData, rsi6, rsi12, sma3, ema6, ema12, atr14, mfi14, adx14, adx20, mom1, mom3, cci12, cci20, rocr3, rocr12, outMACD, outMACDSignal, willR, tsf10, tsf20, trix, bupper, bmiddle, blower))


 
stocks = ['AAPL']
start = datetime.date(2016,1,1)
end = datetime.date(2016,3,31)
rawData = web.DataReader(stocks, 'yahoo',start, end)
rawData = rawData.to_frame()

