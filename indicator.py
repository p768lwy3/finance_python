""" 
#
# Basic Finance Indicator Python Script...
# This is not finished, which is the python script of TA for stock.
# Tool for Stock Analysis
# Function:
#  1. Adjusted: calculate adjusted value of a column. (Not Finished)
#  2. Standardize: calculate (x - mux) / stdx
#  3. ROC
#  4. LogReturn
#  5. MovingAverage(price, windows, method='SMA')
#  6. RSI
#
"""
from datetime import date, datetime, timedelta
from deap import creator, base, tools, algorithms

import random, os, sys, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Indicator(object):
    def __init__(self, ds):
        self.ds = ds

    def Adjusted(self, output_col='Adj{0}', target_col='Close'):
        # self.ds[output_col.format(target_col)]
        pass

    def ColDelta(self, n=1, output_col='{0}Delta', target_col='AdjClose'):
        self.ds[output_col.format(target_col)] = self.ds[target_col].diff().shift(-n)
        return

    def Standardize(self, output_col='Std{0}', target_col='AdjClose'):
        mu = self.ds[target_col].mean()
        sigma = self.ds[target_col].std()
        self.ds[output_col.format(target_col)} = (self.ds[target_col] - mu) / sigma
        return

    def ROC(self, n=1, output_col='roc', target_col='AdjClose'): # rate of return
        self.ds[output_col] = self.ds[target_col].pct_change(n)
        return

    # def LogReturn(...) <- or merge this to roc?
  
    def SMA(self, n=10, output_col='{0}dSMA', target_col='AdjClose'):
        # simple moving avergae
        self.ds[output_col.format(n)] = self.ds[target_col].rolling(window=n).mean()
        return

    def RSI(self, n=1, maType='SMA', outputcol='{0}dRSI', target_col='AdjClose'):
        # relative strength index
        delta = self.ds[target_col].diff()
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
        RolUp = dUp.rolling(n).mean()
        RolDown = dDown.rolling(n).mean().abs()
        RS = RolUp / RolDown
        self.ds[output_col.format(n)] = 100.0 - (100.0 / (1.0 + RS))
        return 

    def FitnessFunction(self, narray=np.array([1,1,1,1], target_col='AdjClose'):
        print('  narray is %s.' % str(xx))
        print('  Check whether there are the columns.')
        print('  Compute them if not exist.')
        print('\n')
        assert narray[1] < narray[2], "short MA values cannot be greater than long MA values."
        if not 'roc' in self.ds.columns:
            self.ROC(n=narray[0], target_col=target_col)
        if not '{0}dSMA'.format(narray[1]) in self.ds.columns:
            self.SMA(n=narray[1], target_col=target_col)
        if not '{0}dSMA'.format(narray[2]) in self.ds.columns:
            self.SMA(n=narray[2], target_col=target_col)
        #if not any('dRSI' in s for s in self.ds.columns):
        if not '{0}dRSI'.format(narray[3]) in self.ds.columns:
            self.RSI(n=narray[3], maType='SMA', target_col=target_col)
        print('  Finish.')

        aa = self.ds[[target_col, 'roc', '{0}dSMA'.format(narray[1]), 
                      '{0}dSMA'.format(narray[2]), '{0}dRSI'.format(narray[3])]].values

    # Not finished
    """
        yyyy = 9999; mm = 12; dd = 31
        isData = aa[index(aa) < datetime(yyyy, mm, dd)]
    
        posBuySignal = isData['rsi'] <= (1 - xx[2]) and isData['smas'] > isData['smal']
        if len(posBuySignal) == 0:
            posBuySignal = None
        posSellSignal = isData['rsi'] > xx[2] and isData['smas'] < isData['smal']
        if len(posSellSignal) == 0:
            posSellSignal = None
        allSignals = np.array(posBuySignal, posSellSignal)
　　      allSignals = allSignals[which(allSignals <= nrow(sampleData))]

        if (!is.null(allSignals) && length(allSignals) >= 50)
            theStat <- SharpeRatio.annualized(isData[sort(allSignals),"rtn"])
        if (is.null(allSignals) | length(allSignals) < 50)
            theStat <- 0  

        return theStat

    def tradingStatistics(isOrOos = TRUE, xx = c(1,1,1,1)):
        print('  xx is %s.' % str(xx))
        rtn = self.ROC(n=1)
        rsi = self.RSI(n=xx[1], maType='SMA')
        smas = self.SMA(n=xx[3])
        smal = self.SMA(n=xx[4])
        aa = np.array(self.ds['Close'], rtn, rsi, smas, smal)
 
        if isOrOos == TRUE:
            sampleData = aa[index(aa) < datetime(2011, 01, 01)]
        else:
            sampleData = aa[index(aa) >= datetime(2011, 01, 01)]

        posBuySignal = isData['rsi'] <= (1 - xx[2]) and isData['smas'] > isData['smal']
        if len(posBuySignal) == 0:
            posBuySignal = None
            posSellSignal = isData['rsi'] > xx[2] and isData['smas'] < isData['smal']
        if len(posSellSignal) == 0:
            posSellSignal = None

        allSignals = np.array(posBuySignal, posSellSignal)
　　      allSignals = allSignals[which(allSignals <= nrow(sampleData))]
  
        totalRtn = sum(sampleData[sort(allSignals),"rtn"])
        numberOfTrades = len(sampleData[sort(allSignals),"rtn"])
        hitRatio = len(which(sampleData[sort(allSignals),"rtn"] > 0))/numberOfTrades

        return [totalRtn, numberOfTrades, hitRatio]

    def RawStochasticValue(...):
    def BollingerBands(...):
    def MACD(...):
    def BiasRatio(...):
    def candlestick(col):
    """

# plot candlestick
# plot day summary(?)

