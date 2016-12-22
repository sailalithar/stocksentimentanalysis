# -*- coding: utf-8 -*-
"""
"""

import pandas as pd, talib as tb, numpy as np


class CreateXMatrix:
    
    def getX(self, rawData, rawDataSnP):
    
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
        
        # get RSI6
        rsi6 = tb.RSI(adjCloseData, 6)
        rsi6SnP = tb.RSI(adjCloseDataSnP, 6)    
        
        # get RSI12
        rsi12 = tb.RSI(adjCloseData, 12)
        rsi12SnP = tb.RSI(adjCloseDataSnP, 12)
        
        # get SMA3
        sma3 = tb.SMA(adjCloseData, 3)
        sma3SnP = tb.SMA(adjCloseDataSnP, 3)
        
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
        
        df_27 = pd.DataFrame(
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
                            
                           }
                        )   
        df_54 = pd.DataFrame(
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
        return df_27, df_54
