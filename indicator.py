""" Basic Finance Indicator Python Script..."""
from datetime import date, datetime, timedelta
from deap import creator, base, tools, algorithms

import random, os, sys, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Indicator(object):
  def __init__(self, ds):
    self.ds = None

  def ROC(self, n=1): # rate of return
    return
  
  def SMA(self, n=1):
    return

  def RSI(self, n=1, maType='SMA'):
    return 

  def FitnessFunction(self, xx=np.array([1,1,1,1]):
    print('  xx is %s.' % str(xx))
    rtn = self.ROC(n=1)
    rsi = self.RSI(n=xx[1], maType='SMA')
    smas = self.SMA(n=xx[3])
    smal = self.SMA(n=xx[4])
    aa = np.array(self.ds['Close'], rtn, rsi, smas, smal)

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
    
def main():
  pass

if __name__ == '__main__':
  main()    
