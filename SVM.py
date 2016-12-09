# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 10:17:19 2016

@author: Shruti
"""
import pandas as pd, pandas.io.data as web,  numpy as np, csv
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from plot_graph import PlotGraph
from x_matrix import CreateXMatrix


class SupportVectorMachine:

    
    #Implement Support Vector Machine
    def RBF_SVM(self,X,y,count,stock):
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
        print("The best parameters are %s with a score of %0.2f print for %d features of %s stock"
          % (grid.best_params_, grid.best_score_, X_train.shape[1], stock))
        
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

    #Implement Tree Algorithm
    def feat_select(self,X,y):
        #feature selection using extermely randomized tree algorithm 
        modelER = ExtraTreesClassifier(n_estimators=250, random_state=0)
        modelER.fit(X, y)    
        #print X
        importances = modelER.feature_importances_
        indices = np.argsort(importances)[-16:] if X.shape[1]==54 else np.argsort(importances)[-8:]
        #print indices
        #print values
        X = X[:,indices]
        return X, indices
        
    # writing accuracy and the relations between the selected features    
    def writeFeatAcc(self, dict_16):
        with open('Select Feature Accuracy SVM.csv', 'w') as csvfile:
            fieldnames = ['Stocks', 'Selected Features', 'Train Accuracy', 'Test Accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
   
            for item in dict_16:
                writer.writerow({'Stocks': item['stock'], 'Selected Features': item['feat_list'], 'Train Accuracy': item['train_acc'], 'Test Accuracy': item['test_acc']})    


    def evaluate(self, stocks, start , end):
        bp = PlotGraph()
        excelDF = []
        count = 0
        # for every catgeory in stocks
        for stockSet in stocks:
            
            all_stock = []
            stock_SnP = []
            all_stock_ER = []
            stock_SnP_ER = []
            stockCat = stockSet.keys()[1]
            stockCodes = stockSet.values()[1] 
            col = stockSet['color']
            
            # for every stock in the category
            for stock in stockCodes:
                rawData = web.DataReader([stock], 'yahoo',start, end)
                rawData = rawData.to_frame()
    			
    			#S&P
                rawDataSnP = (web.DataReader(['^GSPC'], 'yahoo',start, end)).to_frame()
    		
    			#get X matrix
                df_27, df_54 = CreateXMatrix().getX(rawData, rawDataSnP)
                 			
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
       
    		    #Build Y matrix 21 is location of sma columns
                sma3 = X_new_27[:,21]
                Y_27 = [1 if (sma3[x] - sma3[x+3])<0 else -1 for x in range(0,len(sma3)-3)]
                
                sma31 = X_new_54[:,42] # 42 is location of sma columns
                Y_54 = [1 if (sma31[x] - sma31[x+3])<0 else -1 for x in range(0,len(sma31)-3)]
                
                X_new_27 = X_new_27[:-3,:]
                X_new_54 = X_new_54[:-3,:]    
    
    
                X_select_8, feat_8 = self.feat_select(X_new_27,Y_27)    
                X_select_16, feat_16 = self.feat_select(X_new_54,Y_54)
                
                       
                #get Logistic regression
                model_27, y_test_27, train_acc_27, test_acc_27 = self.RBF_SVM(X_new_27,Y_27,count,stock)
                model_54, y_test_54, train_acc_54, test_acc_54 = self.RBF_SVM(X_new_54,Y_54,count,stock)
    
                model_8, y_test_8, train_acc_8, test_acc_8 = self.RBF_SVM(X_select_8,Y_27, count,stock)
                model_16, y_test_16, train_acc_16, test_acc_16 = self.RBF_SVM(X_select_16,Y_54, count,stock)
                
                all_stock.append(
    					{
    						'stock': stock,
    						'model': model_27,
    						'train_acc': train_acc_27,
    						'test_acc': test_acc_27
    					}
    				)
                stock_SnP.append(
    					{
    						'stock': stock,
    						'model': model_54,
    						'train_acc': train_acc_54,
    						'test_acc': test_acc_54
    					}
    				)
          
                all_stock_ER.append(
    					{
    						'stock': stock,
    						'model': model_8,
    						'train_acc': train_acc_8,
    						'test_acc': test_acc_8
    					}
    				)
                stock_SnP_ER.append(
    						{
    							'stock': stock,
    							'model': model_16,
    							'train_acc': train_acc_16,
    							'test_acc': test_acc_16,
                                        'feat_list': feat_16
    						}
        			)
                # creating excel sheet for all stocks 
                excelDF.append({'Stock Category':stockCat,'Stock Name':stock,'LR 27 train':train_acc_27, 'LR 27 test':test_acc_27, 'LR 54 train':train_acc_54, 'LR 54 test':test_acc_54, 'LR 8 train':train_acc_8, 'LR 8 test':test_acc_8, 'LR 16 train':train_acc_16, 'LR 16 test':test_acc_16})
            
            self.writeFeatAcc(stock_SnP_ER)
            print stockCat
        
            # plotting stock for graph per category per algorithm
            folderPath = stockCat+'/SVM/AllStocks/'
            bp.getBarPlot(stockCat, all_stock, col, 27, folderPath)
            folderPath = stockCat+'/SVM/StockSnP/'
            bp.getBarPlot(stockCat, stock_SnP, col, 54, folderPath)
            folderPath = stockCat+'/SVM_ER/AllStocks/'        
            bp.getBarPlot(stockCat, all_stock_ER, col, 8, folderPath)
            folderPath = stockCat+'/SVM_ER/StockSnP/'        
            bp.getBarPlot(stockCat, stock_SnP_ER, col, 16, folderPath)
            
            
        return excelDF 
