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
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
#import matplotlib.pylab as plt
#import plotly.plotly as py

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

def feat_select(X,y):
    
    #feature selection using extermely randomized tree algorithm 
    model = ExtraTreesClassifier(n_estimators=300, random_state=0)
    model.fit(X, y)    
    #print X
    importances = model.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
    #indices = np.argsort(importances)[::-1]
    indices = np.argsort(importances)[-16:]
    values = importances[indices]
    #print indices
    #print values
    X = X[:,indices]
    return X,indices,values    

def RBF_SVM(X,y,count):
    
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    #cross-validate time series data samples    
    cv1 = TimeSeriesSplit(n_splits=2)    
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv1)
    train = int(X.shape[0] * 0.8)
    test = train + 1
    X_train = X[:train,:]
    X_test = X[test:,:]
    y_train = y[:train]
    y_test = y[test:]
    grid.fit(X_train, y_train)
    print count
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    
    # predict class labels for the training set
    predicted_train = grid.predict(X_train)

    # predict class labels for the test set
    predicted_test = grid.predict(X_test)
    acc_train = metrics.accuracy_score(y_train, predicted_train)
    print acc_train

    #print "Accuracy for test set using RBF SVM"
    acc_test = metrics.accuracy_score(y_test, predicted_test)
    print acc_test
    return grid,y_test, acc_train, acc_test
    
def evaluate(stocks, start, end):
    finalList = []
    indicesList = []
    ind_full = np.zeros([16,88],dtype=np.int)
    val_full = np.zeros([16,88])
    train_acc_full = np.zeros([88,1])
    test_acc_full = np.zeros([88,1])
    stock_list = np.zeros([88,1],dtype=object)
    count=0;
    #finalListSnP = []
    for stock in stocks:
        rawData = web.DataReader([stock], 'yahoo',start, end)
        #print("xx"+ str(stock))
        rawData = rawData.to_frame()
        
        #S&P
        rawDataSnP = (web.DataReader(['^GSPC'], 'yahoo',start, end)).to_frame()
    
        #get X matrix
        df = getX(rawData, rawDataSnP)
        #print("xxx"+ str(stock))
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
        sma3 = X_new[:,42]
        Y = [1 if (sma3[x] - sma3[x+3])<0 else -1 for x in range(0,len(sma3)-3)]
        Y1 = [1 if (sma3[x] - sma3[x+3])<-0.03
               else 2 if ((sma3[x] - sma3[x+3])<-0.01 and (sma3[x] - sma3[x+3])>-0.03)
               else 3 if ((sma3[x] - sma3[x+3])>-0.01 and (sma3[x] - sma3[x+3])<0.01)
               else 4 if ((sma3[x] - sma3[x+3])>0.01 and (sma3[x] - sma3[x+3])<0.03)
               else 5 for x in range(0,len(sma3)-3)]        
        #Y1 = [(sma3[x] - sma3[x+3]) for x in range(0,len(sma3)-3)]
        #plt.plot(Y1)
        #Y = Y1
        #Y=Y+[1,1,1]
        #sma3SnP = X_new[:,32]
        #YSnP = [1 if (sma3SnP[x] - sma3SnP[x+3])<0 else -1 for x in range(0,len(sma3SnP)-3)]
        
        X_new = X_new[:-3,:]    
        
        #scores = feat_select(X_new,Y)    
        #get Logistic regression   
        
        X, indices, values = feat_select(X_new,Y)
        #model2,y_test, train_acc, test_acc = RBF_SVM(X_new,Y,count)
        #apply LR for selected features
        model2,y_test, train_acc, test_acc = RBF_SVM(X,Y,count)
        #print grid
        #model , train_acc , test_acc = RBF_SVM(X,Y)
        
        #model2,y_test, train_acc, test_acc = Logistic_reg(X,Y)
        
        ind_full[:,count] = indices
        val_full[:,count] = values
        stock_list[count,0] = stock
        train_acc_full[count,0] = train_acc
        test_acc_full[count,0] = test_acc
        count=count+1
        #print count
        #model2SnP, y_testSnP, train_accSnP, test_accSnP = Logistic_reg(X_new[:, 28:],YSnP)
        #values.append(values)        
        indicesList.append(
                        {
                            'stock': stock,
                            'indices': indices,
                            'values' : values
                        }
                    )
                        
        finalList.append(
                        {
                            'stock': stock,
                            'model': model2,
                            'train_acc': train_acc,
                            'test_acc': test_acc
                        }
                    )

#==============================================================================
#         finalListSnP.append(
#                         {
#                             'stockSnP': stock,
#                             'modelSnP': model2SnP,
#                             'train_accSnP': train_accSnP,
#                             'test_accSnP': test_accSnP
#                         }
#                     )
#==============================================================================

    return finalList, Y1, indicesList, ind_full, val_full, df, train_acc_full, test_acc_full       
    
if __name__ == "__main__":
    
    '''
    Capital Goods Companies (Top 10 market capital) (3)
    
       Tesla Motors, Inc.
       PACCAR Inc.
       Illumina, Inc.    
    '''
    capGoodsStock = ['TSLA', 'PCAR', 'ILMN']
    
    '''
    Consumer Non-Durables (3)
        
        The Kraft Heinz Company
        Mondelez International, Inc.
        Monster Beverage Corporation
    
    Consumer Services (22)
    
        Amazon.com, Inc.
        Comcast Corporation
        Starbucks Corporation
        Charter Communications, Inc.
        Costco Wholesale Corporation
        Twenty-First Century Fox, Inc.
        Netflix, Inc.
        Twenty-First Century Fox, Inc.
        Marriott International
        Liberty Global plc
        Ross Stores, Inc.
        DISH Network Corporation
        Reilly Automotive, Inc.
        JD.com, Inc.
        Equinix, Inc.
        Sirius XM Holdings Inc.
        Paychex, Inc.
        Dollar Tree, Inc.
        Expedia, Inc.
        Viacom Inc.
        Ulta Salon, Cosmetics & Fragrance, Inc.
        Fastenal Company    
    '''
    consServNDStock = ['KHC', 'MDLZ', 'MNST', 'AMZN', 'CMCSA', 'SBUX', 'CHTR',
                       'COST', 'FOX', 'NFLX', 'FOXA', 'MAR', 'LBTYA', 'ROST',
                       'DISH', 'ORLY', 'JD', 'EQIX', 'SIRI', 'PAYX', 'DLTR',
                       'EXPE', 'VIA', 'ULTA', 'FAST']

    '''
    Finance (7)
                 
        CME Group Inc.
        TD Ameritrade Holding Corporation
        Fifth Third Bancorp
        Northern Trust Corporation
        T. Rowe Price Group, Inc.
        Interactive Brokers Group, Inc.
        Huntington Bancshares Incorporated
    '''
    finStock = ['CME', 'AMTD', 'FITB', 'NTRS', 'TROW', 'IBKR', 'HBAN']
    
    '''
    Health Care (15)
    
        Amgen Inc.
        Gilead Sciences, Inc.
        Celgene Corporation
        Walgreens Boots Alliance, Inc.
        Biogen Inc.
        Express Scripts Holding Company
        Regeneron Pharmaceuticals, Inc.
        Alexion Pharmaceuticals, Inc.
        Intuitive Surgical, Inc.
        Vertex Pharmaceuticals Incorporated
        Incyte Corporation
        Mylan N.V.
        Shire plc
        BioMarin Pharmaceutical Inc.
        DENTSPLY SIRONA Inc.
    '''
    hCareStock = ['AMGN', 'GILD', 'CELG', 'WBA', 'BIIB', 'ESRX', 'REGN', 'ALXN',
                  'ISRG', 'VRTX', 'INCY', 'MYL', 'SHPG', 'BMRN', 'XRAY' ]
             
    '''
    Miscellaneous (5)
     
        The Priceline Group Inc. 
        PayPal Holdings, Inc.
        eBay Inc.
        Ctrip.com International, Ltd.
        NetEase, Inc.
    '''
    miscStock = ['PCLN', 'PYPL', 'EBAY', 'CTRP', 'NTES']

    '''
    Public Utilities (1)
    
        T-Mobile US, Inc.
    '''
    pubUtilStock = ['TMUS']
    
    '''
    Technology (30)
    
        Apple Inc.
        Google
        Microsoft Corporation
        Facebook, Inc.
        Intel Corporation
        Cisco Systems, Inc.
        QUALCOMM Incorporated
        Texas Instruments Incorporated
        Broadcom Limited
        Adobe Systems Incorporated
        NVIDIA Corporation
        Baidu, Inc.
        Automatic Data Processing, Inc.
        Yahoo! Inc.
        Applied Materials, Inc.
        Cognizant Technology Solutions Corporation
        NXP Semiconductors N.V.
        Intuit Inc.
        Activision Blizzard, Inc
        Electronic Arts Inc.
        Fiserv, Inc.
        Analog Devices, Inc.
        Micron Technology, Inc.
        Western Digital Corporation
        Lam Research Corporation
        Cerner Corporation
        Autodesk, Inc.
        Symantec Corporation
        Linear Technology Corporation
        IHS Markit Ltd.
    '''
    techStock = ['AAPL', 'GOOGL', 'MSFT', 'FB', 'INTC', 'CSCO', 'QCOM', 'TXN',
                 'AVGO', 'ADBE', 'NVDA', 'BIDU', 'ADP', 'YHOO', 'AMAT', 'CTSH',
                 'NXPI', 'INTU', 'ATVI', 'EA', 'FISV', 'ADI', 'MU', 'WDC', 'LRCX',
                 'CERN', 'ADSK', 'SYMC', 'LLTC', 'INFO']

    '''
    Transportation (2)
    
        CSX Corporation
        American Airlines Group, Inc.
    '''
    transStock = ['CSX','AAL']

    #Total Stock List
    #stocks = capGoodsStock + consServNDStock + finStock + hCareStock + miscStock + pubUtilStock + techStock + transStock
    stocks = ['GOOGL']
    
    #Start and end Time
    start = datetime.date(2015,11,1)
    end = datetime.date(2016,7,15)
    
#    rawData2 = (web.DataReader(['^IXIC'], 'yahoo',start, end)).to_frame()    

    finalList, Y, indicesList, indices, values, df, train_acc_full, test_acc_full = evaluate(stocks, start, end)
    
    with open('final.txt', 'w') as fp:
        for item in finalList:
            fp.write("%s\n" % item)
    
    my_xticks = [item['stock'] for item in finalList]
    y = [item['test_acc'] for item in finalList]
    x = [i+1 for i in range(len(my_xticks))]
    x = np.array(x)
    y = np.array(y)
    

    plt.bar(x, y, align='center')
    plt.xticks(x, my_xticks, size='small')    
    plt.show()
    
    #fig = plt.gcf()
    #plot_url = py.plot_mpl(fig, filename='mpl-basic-bar')
