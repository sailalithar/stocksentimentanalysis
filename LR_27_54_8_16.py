import os, pandas as pd, pandas.io.data as web, datetime, talib as tb, matplotlib.pyplot as plt, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

#Bar Plotting
def getBarPlot(stockCat, finalList, col, numFeat):
    my_xticks = [item['stock'] for item in finalList]
    x = np.array([i+1 for i in range(len(my_xticks))])    

    if len(my_xticks)>30:
        fSize = 5
    else:
        fSize = 7

    #Test Accuracy
    y_test = np.array([item['train_acc']*100 for item in finalList])
    plt.bar(x, y_test, align = 'center', color=col)
    plt.xticks(x, my_xticks, fontsize=fSize ,rotation='vertical')
    plt.xlabel(stockCat + ' ('+ str(numFeat) + ' features) ')
    plt.ylabel("Train Accuracy (%)")
    fig = plt.gcf()
    plt.show()
    plt.draw()
    #Save graph plot in train_Result folder
    if not os.path.exists('train_Result_' + str(numFeat) + '_feat'):
        os.makedirs('train_Result_' + str(numFeat) + '_feat')        
    fig.savefig('train_Result_' + str(numFeat) + '_feat/' + stockCat +'_train.png', format='png', bbox_inches='tight', dpi=1000)
    
    
    #Train Accuracy
    y_test = np.array([item['test_acc']*100 for item in finalList])
    plt.bar(x, y_test, align = 'center', color=col)
    plt.xticks(x, my_xticks, fontsize=fSize ,rotation='vertical')
    plt.xlabel(stockCat + ' ('+ str(numFeat) + ' features) ')
    plt.ylabel("Test Accuracy (%)")
    fig = plt.gcf()
    plt.show()
    plt.draw()
    #Save graph plot in test_Result folder
    if not os.path.exists('test_Result_' + str(numFeat) + '_feat'):
        os.makedirs('test_Result_' + str(numFeat) + '_feat')        
    fig.savefig('test_Result_' + str(numFeat) + '_feat/'+stockCat +'_test.png', format='png', bbox_inches='tight', dpi=1000)
    
    
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
    #print "Accuracy for training set using Logistic regression"
    acc_train = metrics.accuracy_score(y_train, predicted_train)
    #print acc_train

    #print "Accuracy for test set using Logistic regression"
    acc_test = metrics.accuracy_score(y_test, predicted_test)
    #print acc_test

    #validation for model
    return model2, y_test, acc_train, acc_test
    
def feat_select(X,y):
    
    #feature selection using extermely randomized tree algorithm 
    model = ExtraTreesClassifier(n_estimators=250, random_state=0)
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
    return X  

def evaluate(stocks, start, end):
    for stockSet in stocks:
        finalList_27 = []
        finalList_54 = []
        finalList_select_8 = []
        finalList_select_16 = []
        stockCat = stockSet.keys()[1]
        stockCodes = stockSet.values()[1] 
        col = stockSet['color']
        for stock in stockCodes:
            rawData = web.DataReader([stock], 'yahoo',start, end)
            rawData = rawData.to_frame()
			
			#S&P
            rawDataSnP = (web.DataReader(['^GSPC'], 'yahoo',start, end)).to_frame()
		
			#get X matrix
            df_27, df_54 = getX(rawData, rawDataSnP)
			
			#convert dataframe to training matrix
            X_27 = pd.DataFrame.as_matrix(df_27)
            X_54 = pd.DataFrame.as_matrix(df_54)
		
			#Normalize X matrix
            for n in range(0,X_27.shape[1]):
                X_27[:,n]=X_27[:,n]/np.nanmax(abs(X_27[:,n]))
			
            for n in range(0,X_54.shape[1]):
                X_54[:,n]=X_54[:,n]/np.nanmax(abs(X_54[:,n]))

			#cut X matrix to remove NaN's
            X_new_27 = X_27[39:,:]
            X_new_54 = X_54[39:,:]
   
		#Build Y matrix
            sma3 = X_new_27[:,21]
            Y_27 = [1 if (sma3[x] - sma3[x+3])<0 else -1 for x in range(0,len(sma3)-3)]
            
            sma3 = X_new_54[:,42]
            Y_54 = [1 if (sma3[x] - sma3[x+3])<0 else -1 for x in range(0,len(sma3)-3)]
            
            X_new_27 = X_new_27[:-3,:]
            X_new_54 = X_new_54[:-3,:]    


            X_select_8 = feat_select(X_new_27,Y_27)    
            X_select_16 = feat_select(X_new_54,Y_54)
            
            #get Logistic regression
            model_27, y_test_27, train_acc_27, test_acc_27 = Logistic_reg(X_new_27,Y_27)
            model_54, y_test_54, train_acc_54, test_acc_54 = Logistic_reg(X_new_54,Y_54)

            model_8, y_test_8, train_acc_8, test_acc_8 = Logistic_reg(X_select_8,Y_27)
            model_16, y_test_16, train_acc_16, test_acc_16 = Logistic_reg(X_select_16,Y_54)
			
            finalList_27.append(
					{
						'stock': stock,
						'model': model_27,
						'train_acc': train_acc_27,
						'test_acc': test_acc_27
					}
				)
            finalList_54.append(
					{
						'stock': stock,
						'model': model_54,
						'train_acc': train_acc_54,
						'test_acc': test_acc_54
					}
				)
      
            finalList_select_8.append(
					{
						'stock': stock,
						'model': model_8,
						'train_acc': train_acc_8,
						'test_acc': test_acc_8
					}
				)
	    finalList_select_16.append(
						{
							'stock': stock,
							'model': model_16,
							'train_acc': train_acc_16,
							'test_acc': test_acc_16
						}
					)
        print stockCat
        getBarPlot(stockCat, finalList_27, col, 27)
        getBarPlot(stockCat, finalList_54, col, 54)
        getBarPlot(stockCat, finalList_select_8, col, 8)
        getBarPlot(stockCat, finalList_select_16, col, 16)
    return finalList_27, finalList_54, finalList_select_8, finalList_select_16 
    
if __name__ == "__main__":
    
    '''
    Capital Goods Companies (Top 10 market capital) (3)
    
       Tesla Motors, Inc.
       PACCAR Inc.
       Illumina, Inc.    
    '''
    capGoodsStock = {'Capital Goods': ['TSLA', 'PCAR', 'ILMN'], 'color': '#00FFFF'}
    
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
    consServNDStock = { 'Consumer Services':
                        ['KHC', 'MDLZ', 'MNST', 'AMZN', 'CMCSA', 'SBUX', 'CHTR',
                       'COST', 'FOX', 'NFLX', 'FOXA', 'MAR', 'LBTYA', 'ROST',
                       'DISH', 'ORLY', 'JD', 'EQIX', 'SIRI', 'PAYX', 'DLTR',
                       'EXPE', 'VIA', 'ULTA', 'FAST'],
                       'color': '#0080FF'                       
                      }

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
    finStock = { 'Finance':
                ['CME', 'AMTD', 'FITB', 'NTRS', 'TROW', 'IBKR', 'HBAN'],
                'color': '#7F00FF'}
    
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
    hCareStock = { 'Health Care':
                    ['AMGN', 'GILD', 'CELG', 'WBA', 'BIIB', 'ESRX', 'REGN', 'ALXN',
                     'ISRG', 'VRTX', 'INCY', 'MYL', 'SHPG', 'BMRN', 'XRAY' ],
                     'color': '#00FF80'
                 }             
    '''
    Miscellaneous (5)
     
        The Priceline Group Inc. 
        PayPal Holdings, Inc.
        eBay Inc.
        Ctrip.com International, Ltd.
        NetEase, Inc.
    '''
    miscStock = {'Miscellaneous':
                    ['PCLN', 'PYPL', 'EBAY', 'CTRP', 'NTES'],
                 'color': '#FF8000'
                }

    '''
    Public Utilities (1)
    
        T-Mobile US, Inc.
    '''
    pubUtilStock = {'Public Utilities': ['TMUS'], 'color':'#808080'}    

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
    techStock = { 'Technology':
                    ['AAPL', 'GOOGL', 'MSFT', 'FB', 'INTC', 'CSCO', 'QCOM', 'TXN',
                 'AVGO', 'ADBE', 'NVDA', 'BIDU', 'ADP', 'YHOO', 'AMAT', 'CTSH',
                 'NXPI', 'INTU', 'ATVI', 'EA', 'FISV', 'ADI', 'MU', 'WDC', 'LRCX',
                 'CERN', 'ADSK', 'SYMC', 'LLTC', 'INFO'],
                 'color' : '#994C00'
                }

    '''
    Transportation (2)
    
        CSX Corporation
        American Airlines Group, Inc.
    '''
    transStock = {'Transportation': ['CSX','AAL'], 'color': '#660000' }

    #Total Stock List
    stocks = [capGoodsStock, consServNDStock, finStock, hCareStock, miscStock, pubUtilStock, techStock, transStock]
    
    #Start and end Time
    startYear = 2015
    startMonth = 11
    startDay = 1
    
    endYear = 2016
    endMonth = 7
    endDay = 15
    
    startTime = datetime.date(startYear, startMonth, startDay)
    endTime = datetime.date(endYear, endMonth, endDay)

    finalList_27, finalList_54, finalList_select_8, finalList_select_16 = evaluate(stocks, startTime, endTime)
    
    with open('finalList_27.txt', 'w') as fp:
        for item in finalList_27:
            fp.write("%s\n" % item)
    
    with open('finalList_54.txt', 'w') as fp:
        for item in finalList_54:
            fp.write("%s\n" % item)
    
    with open('finalList_select_8.txt', 'w') as fp:
        for item in finalList_select_8:
            fp.write("%s\n" % item)
    
    with open('finalList_select_16.txt', 'w') as fp:
        for item in finalList_select_16:
            fp.write("%s\n" % item)
