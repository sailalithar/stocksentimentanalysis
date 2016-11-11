# -*- coding: utf-8 -*-
"""
Created on Sun Nov 06 21:47:55 2016

@author: saila
"""
import os
import pandas as pd
import datetime
import pandas.io.data as web
import matplotlib.pyplot as plt

def plot_selected(df,columns,start_date,end_date):
    plot_data(df.ix[start_date:end_date,columns],title="Share Prices")
   # plt.show()

def symbol_to_path(symbol, base_dir="C:\Users\saila\Desktop\ML\Project\Stocksentimentanalysis\Data"):
    #print os.path.join(base_dir,"{}.csv".format(str(symbol)))
    return os.path.join(base_dir,"{}.csv".format(str(symbol)))

def get_data(symbols,start,end):
    
        #stocks = ['ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
        ls_key = 'Adj Close'
        #start = datetime(2014,1,1)
        #end = datetime(2014,3,28)    
        f = web.DataReader(symbols, 'yahoo',start, end)
        cleanData = f.ix[ls_key]
        dataFrame = pd.DataFrame(cleanData)
        dataFrame.reset_index(inplace=True,drop=False)
        return dataFrame

def compute_daily_returns(df):
     
    daily_returns=df.copy()
    daily_returns[1:]=(df[1:]/df[:-1].values) -1
    daily_returns[0,:]=0
    return daily_returns
 
def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    r_mean = pd.Series.to_frame(pd.rolling_mean(values, window=window))    
    return r_mean

def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    r_std = pd.Series.to_frame(pd.rolling_std(values,window=window))    
    return r_std

def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band=rm+2*rstd
    lower_band=rm-2*rstd
    return upper_band, lower_band

def test_run():
    
    dates = pd.date_range('2016-01-01', '2016-01-31')
    start = datetime.date(2016,1,1)
    end = datetime.date(2016,1,31)
    symbols = ['ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
    df = get_data(symbols, start, end)
    #df_SPY = get_data(['SPY'], start, end)
    s = get_data(['SPY'], start, end)
    s = get_rolling_mean(s['SPY'],2)
    s = s.rename(columns = {'SPY':'SPY'+"_Mean"})
    x = get_data(['SPY'], start, end)
    x = get_rolling_std(x['SPY'],2)
    x = x.rename(columns = {'SPY':'SPY'+"_Std"})
    #r_mean=pd.DataFrame(index=dates)
    #s = get_rolling_mean(df_SPY,2)
    for symbol in symbols:
        r=get_rolling_mean(df[symbol],2)
        r = r.rename(columns = {symbol:symbol+"_Mean"})
        #s = pd.Series.to_frame(s)
        #print r
        s=s.join(r)
        p = get_rolling_std(df[symbol],2)
        p = p.rename(columns = {symbol:symbol+"_Std"})
        #r_mean
        x=x.join(p)
    print s
    print x
    #print df
   # plot_data(df)

def normalize_data(df):
    return df/df.ix[0,:]
   
def plot_data(df,title="Stock prices", xlabel="Date", ylabel="Price"):
    ax=df.plot(title=title,fontsize=2)    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()   
    
if __name__ == "__main__":
    test_run()