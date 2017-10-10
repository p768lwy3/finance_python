"""
This is not finished, which is the python script of TA for stock.
Tool for Stock Analysis
Function:
  AdjClose: calculate adjusted price. (Not Finished)
  StandardizePrice
  DailyReturn
  LogReturn
  MovingAverage(price, windows, method='SMA')
  RSI
"""
# Import:
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates
from matplotlib.dates import  DateFormatter, WeekdayLocator, HourLocator, DayLocator, MONDAY
from matplotlib.finance import candlestick, plot_day_summary, candlestick2
from pylab import *
from sklearn.cross_validation import train_test_split
from yahoo_finance import Share

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web

def AdjClose():
  """Code: """
  pass

def PriceDelta(price, windows=1):
  delta = np.array([0 for i in range(windows)])
  for i in range(windows, len(price)):
    value = price[i] - price[i-windows]
    delta = np.append(delta, value)
  return delta

def StandardizePrice(price):
  if type(price) == list:
    price = np.array(price)
  stdprice = (price - price.mean())/price.std()
  return stdprice

def DailyReturn(price):
  dailyreturn = []
  for i in range(0, len(price)):
    dailyreturn.append(0) if i == 0 else dailyreturn.append((float(price[i]) - float(price[i-1]))/float(price[i-1]))
  return dailyreturn

def LogReturn(price):
  logreturn = []
  for i in range(0, len(price)):
    logreturn.append(0) if i == 0 else logreturn.append(np.log(float(price[i])) - np.log(float(price[i-1])))
  return logreturn

def MovingAverage(price, windows, method='SMA'):
  if type(price) == list:
    price = np.array(price)
  movavg = np.array([np.nan for h in range(0, windows)])
  if(method == 'SMA'):
    """Simple Moving Average: summation(i = 1, period) price / period"""
    counter = 0
    for i in range(windows, len(price)):
      value = price[counter:i+counter].mean()
      movavg = np.append(movavg, value)
      counter += 1

  elif(method == 'EMA'):
    # SMA: 10 period sum / 10 
    # Multiplier: (2 / (Time periods + 1) ) = (2 / (10 + 1) ) = 0.1818 (18.18%)
    # EMA: {Close - EMA(previous day)} x multiplier + EMA(previous day). 
    movavg = np.array([np.nan for h in range(0, windows-1)])
    movavg = np.append(movavg, price[:windows].mean())
    parameter = 2 /(windows + 1)
    for i in range(windows + 1, len(price)):
      value = (price[i] - movavg[i-1]) * parameter + movavg[i-1]
      movavg = np.append(movavg, value)

  elif(method == 'WMA'):
    # WMA = summation(i=1, windows) i * price / sumof(1, windows)
    base = (1 + windows) * windows / 2
    for i in range(windows, len(price)):
      parameter = 1
      for j in range(i-windows, i):
        value += price[j] * parameter / base
        parameter += 1
      movavg = np.append(movavg, value)
      value = 0
  return movavg

def RSI(price, windows = 14, method = 'SMA'):
  if type(price) == list:
    price = np.array(price)
  delta = PriceDelta(price)
  up, down = delta.copy(), delta.copy()
  up[up < 0] = 0
  down[down > 0] = 0
  if(method == 'SMA'):
    RolUp = MovingAverage(up, windows)
    RolDown = np.abs(MovingAverage(down, windows))
  elif(method == 'EMA'):
    com = (windows - 1) / 2
    RolUp = MovingAverage(up, windows, method='EMA')
    RolDown = np.abs(MovingAverage(down, windows, method='EMA'))
  RS = RolUp.astype('float') / RolDown.astype('float')
  Rsi = 100.0 - (100.0 / (1.0 + RS))
  return Rsi

class StochasticOscillator(object): # %K%D
  def __init__(self):
    """
      https://zh.wikipedia.org/zh-hk/%E9%9A%8F%E6%9C%BA%E6%8C%87%E6%A0%87
    """
    pass

  def RawStochasticValue(self, price_close, price_high, price_low, period_n=9):
    """ Rsv = Cn - Ln/ Hn-Ln """
    for i in range(
    Rsv = np.array([])
    for i in range(len(price_close)):
      if i < period_n:
        highest = price_high[:i].max()
        lowest = price_low[:i].min()
      elif i >= period_n and len(price_close)-i >= period_n:
        highest = price_high[i-period_n:i].max()
        lowest = price_low[i-period_n:i].min()
      else:
        highest = price_high[i:].max()
        lowest = price_low[i:].min()
      value = (price_close[i] - lowest)/(highest - lowest)
      Rsv = np.append(Rsv, value)
    return Rsv

  def percentage_k(self, price_close, price_high, price_low, period_n=9, alpha=1/3):
    Rsv = self.RawStochasticValue(price_close, price_high, price_low, period_n=9)
    per_k = [0.5]
    per_k.extend([alpha*rsv[i] + (1-alpha)*per_k[i-1] for i in range(1, len(Rsv))])
    return per_k

  def percentage_d(self, price_close, price_high, price_low, period_n=9, alpha=1/3):
    PerK = self.percentage_k(price_close, price_high, price_low, period_n=9, alpha=1/3)
    per_d = [0.5]
    per_d.extend([alpha*per_k[i] + (1-alpha)*per_d[i-1] for i in range(1, len(per_k))])
    return per_d

  def signal(self, price_close, price_high, price_low, period_n=9, alpha=1/3):
    PerK = self.percentage_k(price_close, price_high, price_low, period_n=9, alpha)
    PerD = self.percentage_D(price_close, price_high, price_low, period_n=9, alpha)
    signallist = PerK - PerD
    return ['up' if PerK[i] - PerD[i] >= 0 else 'down' for i in range(len(PerK))]

class BollingerBands(object):
  def __init__(self, period_n=20, std_k=2):
    """https://zh.wikipedia.org/zh-hk/%E5%B8%83%E6%9E%97%E5%B8%A6"""
    pass
  def middleband(self):
    pass
  def upperband(self):
    return self.middleband() + k * price[i:].std()
  def lowerband(self):
    middle = middleband()

class MACD(object):
  def __init__(self, period_1=12, period_2=26):
    """https://zh.wikipedia.org/zh-hk/MACD"""
    pass
  def DIFValue(object):
    """dif = EMA(close,12) - EMA(close,26)"""
    pass
  def DEMValue(object, period_dem=9):
    """dem = ema(dif,9)"""
    pass
  def OSCValue(self):
    """osc = dif-dem"""
    pass
  def hist(self):
    #Plot.hist()
    pass

class BiasRatio(object):
  def __init__(self, price_close):
    """https://zh.wikipedia.org/zh-hk/%E4%B9%96%E9%9B%A2%E7%8E%87"""
    pass
  def nBIAS(self, period_n=3):
    """nbias = (close - MAn) / MAn"""
    pass

  def maBIAS(self, period_short, period_long):
    """maBIAS = (MAshort - MAlong)/MAlong
    return (MAshort - MAlong)/MAlong"""
    pass

  def nmBIAS(self, period_short=3, period_long=6):
    """nmBIAS = nBIAS-mBIAS"""
    '''return nBIAS(period_n=period_short) - nBIAS(period_n=period_long)'''
    pass

class (object):
  def __init__(self):
    """"""
    pass

class (object):
  def __init__(self):
    """"""
    pass

class Plot(object):
  def __init__(self, dataframe):
    self.dataset = dataframe

  '''
  def candlestick_chart(price_open, price_close, price_high, price_low):
    """https://zh.wikipedia.org/zh-hk/K%E7%BA%BF"""
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays    = DayLocator()              # minor ticks on the days
    weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    dayFormatter = DateFormatter('%d')      # e.g., 12
    #starting from dates expressed as strings...
    Date1 = '01/01/2010'
    Date2 = '02/01/2010'
    #...you convert them in float numbers....
    Date1 = date2num(datetime.strptime(Date1, "%d/%m/%Y"))
    Date2 = date2num(datetime.strptime(Date2, "%d/%m/%Y"))
    #so redefining the Prices list of tuples...
    Prices = [(Date1, 1.123, 1.212, 1.463, 1.056), (Date2,1.121, 1.216, 1.498, 1.002)]
    #and then following the official example. 
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)
    candlestick(ax, Prices, width=0.6)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
  def hist(self):
    pass
  '''

def main():
  pass
  
if __name__ == "__main__":
  main()
