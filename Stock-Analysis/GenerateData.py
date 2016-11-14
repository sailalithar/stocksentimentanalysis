# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:34:55 2016

@author: Shruti Kamtekar
"""
import math
import numpy as np
import talib as tb


class GenerateData:
  def NormalizeX(self,feature):
      nanFeature =  [0 if math.isnan(x) else x for x in range(len(feature))]
      fMean = np.mean(nanFeature)
      normalFeature = [((fMean-x)/fMean) for x in range(len(nanFeature))]
      return normalFeature

  def GetData(self,rawData):
  	#getting raw data
	highData = np.array(rawData['High'], dtype=np.dtype(float))
	lowData = np.array(rawData['Low'], dtype=np.dtype(float))
	closeData = np.array(rawData['Close'], dtype=np.dtype(float))
	openData = np.array(rawData['Open'], dtype=np.dtype(float))
	        
	# get Adj Close
	adjCloseData = np.array(rawData['Adj Close'], dtype=np.dtype(float))
	adjCloseData = np.array(self.NormalizeX(adjCloseData), dtype=np.dtype(float))
	    
	# get VolumeData
	volumeData = np.array(rawData['Volume'], dtype=np.dtype(float))
	volumeData = np.array(self.NormalizeX(volumeData), dtype=np.dtype(float))
	    
	# OBV
	obv = tb.OBV(adjCloseData,volumeData)
	obv = np.array(self.NormalizeX(obv), dtype=np.dtype(float))
	    
	# get RSI6
	rsi6 = tb.RSI(adjCloseData, 6)
	rsi6 = np.array(self.NormalizeX(rsi6), dtype=np.dtype(float))
	    
	# get RSI12
	rsi12 = tb.RSI(adjCloseData, 12)
	rsi12 = np.array(self.NormalizeX(rsi12), dtype=np.dtype(float))
	    
	# get SMA3
	sma3 = tb.SMA(adjCloseData,3)
	sma3 = np.array(self.NormalizeX(sma3), dtype=np.dtype(float))
	    
	# get EMA6
	ema6 = tb.EMA(adjCloseData, 6)
	ema6 = np.array(self.NormalizeX(ema6), dtype=np.dtype(float))
	    
	# get EMA12
	ema12 = tb.EMA(adjCloseData, 6)
	ema12 = np.array(self.NormalizeX(ema12), dtype=np.dtype(float))
	    
	# get ATR14
	atr14 = tb.ATR(highData, lowData, closeData, 14)
	atr14 = np.array(self.NormalizeX(atr14), dtype=np.dtype(float))
	    
	# get MFI14
	mfi14 = tb.MFI(highData, lowData, closeData, volumeData, 14)
	mfi14 = np.array(self.NormalizeX(mfi14), dtype=np.dtype(float))
	    
	# get ADX14
	adx14 = tb.ADX(highData, lowData, closeData, 14)
	adx14 = np.array(self.NormalizeX(adx14), dtype=np.dtype(float))
	    
	# get ADX20
	adx20 = tb.ADX(highData, lowData, closeData, 20)
	adx20 = np.array(self.NormalizeX(adx20), dtype=np.dtype(float))
	    
	# get MOM1
	mom1 = tb.MOM(adjCloseData, 1)
	mom1 = np.array(self.NormalizeX(mom1), dtype=np.dtype(float))
	    
	# get MOM3
	mom3 = tb.MOM(adjCloseData, 3)
	mom3 = np.array(self.NormalizeX(mom3), dtype=np.dtype(float))
	    
	# get CCI12
	cci12 = tb.CCI(highData, lowData, closeData, 12)
	cci12 = np.array(self.NormalizeX(cci12), dtype=np.dtype(float))
	    
	# get CCI20
	cci20 = tb.CCI(highData, lowData, closeData, 20)
	cci20 = np.array(self.NormalizeX(cci20), dtype=np.dtype(float))    
	    
	# get ROCR3
	rocr3 = tb.ROCR100(adjCloseData, 3)
	rocr3 = np.array(self.NormalizeX(rocr3), dtype=np.dtype(float))
	    
	# get ROCR12
	rocr12 = tb.ROCR100(adjCloseData, 12)
	rocr12 = np.array(self.NormalizeX(rocr12), dtype=np.dtype(float))
	    
	# get outMACD, outMACDSignal, outMACDHist
	outMACD, outMACDSignal, outMACDHist = tb.MACD(adjCloseData,fastperiod=12,slowperiod=26,signalperiod=9)
	outMACD = np.array(self.NormalizeX(outMACD), dtype=np.dtype(float))
	outMACDHist = np.array(self.NormalizeX(outMACDHist), dtype=np.dtype(float))
	outMACDSignal = np.array(self.NormalizeX(outMACDSignal), dtype=np.dtype(float))

	# get WILLR
	willR = tb.WILLR(highData,lowData,closeData)
	willR = np.array(self.NormalizeX(willR), dtype=np.dtype(float))
	        
	# get TSF10
	tsf10 = tb.TSF(adjCloseData, 10)
	tsf10 = np.array(self.NormalizeX(tsf10), dtype=np.dtype(float))
	    
	# get TSF20
	tsf20 = tb.TSF(adjCloseData, 20)
	tsf20 = np.array(self.NormalizeX(tsf20), dtype=np.dtype(float))
	    
	# get TRIX, default to 12 days
	trix = tb.TRIX(adjCloseData, 12)
	trix = np.array(self.NormalizeX(trix), dtype=np.dtype(float))
	    
	# get BBANDSUPPER, BBANDSMIDDLE, BBANDSLOWER
	bupper, bmiddle, blower = tb.BBANDS(adjCloseData, 10, 2, 2, 0)
	bupper = np.array(self.NormalizeX(bupper), dtype=np.dtype(float))
	bmiddle = np.array(self.NormalizeX(bmiddle), dtype=np.dtype(float))
	blower = np.array(self.NormalizeX(blower), dtype=np.dtype(float))


	# create feature matrix
	X = np.column_stack((adjCloseData, obv, volumeData, rsi6, rsi12, sma3, ema6, ema12, atr14, mfi14, adx14, adx20, mom1, mom3, cci12, cci20, rocr3, rocr12, outMACD, outMACDSignal, willR, tsf10, tsf20, trix, bupper, bmiddle, blower))

	# create class matirx
	y= [1 if adjCloseData[x+1]-adjCloseData[x]>0 else -1 for x in range(len(adjCloseData)-1)]
	y.insert(0,-1)
	y = np.array(y)

	return X,y
	 
	