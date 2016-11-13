import os
import pandas as pd
import datetime
import pandas.io.data as web
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def cleanData(rawData):
	openData = rawData.ix['Open']
	highData = rawData.ix['High']
	lowData = rawData.ix['Low']
	closeData = rawData.ix['Close']
	volumeData = rawData.ix['Volume']
	adjCloseData = rawData.ix['Adj Close']
	
	#WillR
	willR = (highData - closeData)*100/(highData - lowData)

	#ROCR3	
	rocR3 =	adjCloseData.div(adjCloseData.shift(3))

	#ROCR12	
	rocR12 = adjCloseData.div(adjCloseData.shift(12))

	#MOM1
	mom1 = adjCloseData.diff(1)
	
	#MOM3
	mom3 = adjCloseData.diff(3)
	
	#RSI6
	delta = adjCloseData.diff()
	priceU = delta.copy()
	priceD = delta.copy()
	priceU[priceU < 0] = 0
	priceD[priceD > 0] = 0
	avgPriceU = pd.rolling_mean(priceU, 6)
	avgPriceD = pd.rolling_mean(priceD, 6).abs()
	RSI6 = (avgPriceU/(avgPriceU + avgPriceD)) * 100
	
	#RSI12
	delta = adjCloseData.diff()
	priceU = delta.copy()
	priceD = delta.copy()
	priceU[priceU < 0] = 0
	priceD[priceD > 0] = 0
	avgPriceU = pd.rolling_mean(priceU, 12)
	avgPriceD = pd.rolling_mean(priceD, 12).abs()
	RSI12 = (avgPriceU/(avgPriceU + avgPriceD)) * 100

	#CCI12
	TP = (openData+lowData+closeData)/3
	CCI12 = (TP - pd.rolling_mean(TP, 12)) / (0.15 * pd.rolling_std(TP, 12)) 
	
	#CCI20
	TP = (openData+lowData+closeData)/3
	CCI20 = (TP - pd.rolling_mean(TP, 20)) / (0.15 * pd.rolling_std(TP, 20)) 

	#ATR
	
	
	
def get_data(symbols,start,end):
    
        #stocks = ['ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
        #ls_key = 'Adj Close'
        #start = datetime(2014,1,1)
        #end = datetime(2014,3,28)    
        f = web.DataReader(symbols, 'yahoo',start, end)
        #print f
        #print f.to_frame()
#==============================================================================
#         cleanData = f.ix[ls_key]
#         dataFrame = pd.DataFrame(cleanData)
#         dataFrame.reset_index(inplace=True,drop=False)
#==============================================================================
        return f


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

#==============================================================================
# def test_run():
#     
#     #dates = pd.date_range('2016-01-01', '2016-01-31')
#     start = datetime.date(2016,1,1)
#     end = datetime.date(2016,1,31)
#     symbols = ['AAPL']
#     y = [0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1]
#     #symbols = ['ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
#     df = get_data(symbols, start, end)
#     df = df.as_matrix(columns=None)
#     print df
#     XX = df
#     Logistic_reg(df,y)
#     #print df.as_matrix(columns=None)
#     #print df.dtypes
#==============================================================================
def Logistic_reg(X,y):
    clf2 = LogisticRegression().fit(X, y)
    X_new = [[ 5.0,  3.6,  1.3,2.2,4.5,6.6]]
    LogisticRegression(C=1.0, dual=False, fit_intercept=True, intercept_scaling=1,
          penalty='l2', tol=0.0001)
#    print clf2.predict_proba(X_new)

if __name__ == "__main__":
    #test_run()
    start = datetime.date(2000,1,1)
    end = datetime.date(2016,11,1)
    symbols = ['AAPL']
    y = [0,0,1,0,0,0,1,1,1,0,1,0,0,1,1,0,0,1,1]
    #symbols = ['ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
    df = get_data(symbols, start, end)
    print df.to_frame().iloc[0:4]
    cleanData(df)
'''
    df = df.as_matrix(columns=None)
    print df
    #XX = df[:,0:3]
    Logistic_reg(df,y)
'''
