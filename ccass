""" Crawling Data from CCASS """
# Import:

import os, random, requests, time, warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA

HTML_PARSER = 'html.parser'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0'}

def UrlGenerator(head, end, start):
  """Generate url for CCASS"""
  urllist = []
  urlhead = head
  delta = timedelta(days=1)
  while(end >= start):
    tailurl = '{0}-{1}-{2}'.format(end.strftime('%Y'),end.strftime('%m'),end.strftime('%d'))
    urllist.append(urlhead + tailurl)
    end -= delta
  return urllist

def dataCrawling(url):
  temp = []; dict = {}
  list_req = requests.get(url, headers=headers)

  if(list_req.status_code == requests.codes.ok):
    soup = BeautifulSoup(list_req.content, HTML_PARSER)
    table = soup.find_all('table', attrs={'class':'optable'})
    summary = []
    for tr in table[0].find_all('tr'):
      l = []
      for th in tr.find_all('th'):
        l.append(th.text)
      for td in tr.find_all('td'):
        l.append(td.text)
      summary.append(l)

    summarydf = pd.DataFrame.from_records(summary)
    summarydf = summarydf.drop(summarydf.columns[[3]], axis=1)
    summarydf.columns = summarydf.iloc[0]
    summarydf = summarydf.reindex(summarydf.index.drop(0))

    details = []
    for tr in table[1].find_all('tr'):
      l = []
      for th in tr.find_all('th'):
        l.append(th.text)
      for td in tr.find_all('td'):
        l.append(td.text)
      details.append(l)

    detailsdf = pd.DataFrame.from_records(details)
    detailsdf.columns = detailsdf.iloc[0]
    detailsdf = detailsdf.reindex(detailsdf.index.drop(0))
    detailsdf['Row'] = detailsdf['Row'].str.extract('(\d+)', expand=True)
    detailsdf['Stake%'] = detailsdf['Stake%'].str.extract('(\d+[\.]?\d+)', expand=True)
    print(summarydf, detailsdf)
    return summarydf, detailsdf


def Standarize(df):
  df = (df - df.mean())/df.std()
  #df = (df - df.mean())/(df.max() - df.min())
  return df

def PerChange(df):
  df = df.pct_change()
  return df

def correLation(df): 
  """may be no use...
     try to plot some correlation square plot for serval times
  """
  for i in range(0, 3):
    ## randomly pick 20 holders
    l1 = random.sample(range(0, len(df.columns)), 20)
    temp_df = df.iloc[:, l1]
    plt.imshow(temp_df.corr(), cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()
    tick_marks = [i for i in range(len(temp_df.columns))]
    plt.xticks(tick_marks, temp_df.columns, rotation=90.)
    plt.yticks(tick_marks, temp_df.columns)
    plt.show()

def kmeansWithpca(df, plot=False): 
  """this is not a good method...
  """
  ipca = IncrementalPCA(n_components=2).fit(df)
  X = ipca.transform(df)
  kmeans = KMeans(n_clusters=5, random_state=0, init='k-means++', n_init=10).fit(X)
  if plot == True:
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02	# point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n' 'Centroids are marked with white cross')
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    plt.xticks(()); plt.yticks(()); plt.show()
    print(kmeans.labels_)

  return kmeans.labels_, kmeans.cluster_centers_

def plotLine(df):
  # We use seaborn to plot what we have
  ax = None
  ax = sns.tsplot(ax=ax, data=df.values, err_style="unit_traces")
  plt.show()

def hierarchicalClustering(df):
  ## https://stackoverflow.com/questions/34940808/hierarchical-clustering-of-time-series-in-python-scipy-numpy-pandas
  ## try to plot some correlation square plot for serval times.
  for i in range(0, 3):
    ## randomly pick 20 holders
    l1 = random.sample(range(0, len(df.columns)), 40)
    temp_df = df.iloc[:, l1]
    # Just one line :)
    # temp_df = temp_df.T
    # Z = hac.linkage(temp_df, 'single', 'correlation')
    # OR Here we decided to use spearman correlation
    correlation_matrix = temp_df.corr(method='spearman')
    # Do the clustering
    Z = hac.linkage(correlation_matrix, 'single')

    # Plot the dendogram
    plt.figure()
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hac.dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
    plt.show()

def main():
  stockno = 3333
  start = datetime.datetime(2017,5,15)
  end = datetime.datetime(2017,5,30)
  head = 'https://webb-site.com/ccass/choldings.asp?sort=holddn&sc={0}&d='.format(str(stockno))	

  df0, df1 = dataCrawling('https://webb-site.com/ccass/choldings.asp?sort=holddn&sc={0}&d=2017-05-31'.format(str(stockno)))
  df0.index = df0['Type of holder']
  summarydf = pd.DataFrame(columns=df0['Type of holder'])
  summarydf = summarydf.append(df0['Holding'])

  ## drop all '' in CCASS ID:
  df1 = df1[df1['CCASS ID'] != '']
  df1.index = df1['CCASS ID']
  detailsdf = pd.DataFrame(columns=df1['CCASS ID'])
  detailsdf = detailsdf.append(df1['Holding'])

  l0 = urlGenerator(head, end, start)
  for url in l0:
    summarydf0, detailsdf0 = dataCrawling(url)
    summarydf0.index = summarydf0['Type of holder']
    summarydf = summarydf.append(summarydf0['Holding'])
    detailsdf0 = detailsdf0[detailsdf0['CCASS ID'] != '']
    detailsdf0.index = detailsdf0['CCASS ID']
    detailsdf = detailsdf.append(detailsdf0['Holding'])

    idx = pd.date_range(start, end + datetime.timedelta(days=1))[::-1]
    '''summarydf.index = idx
    summarydf = summarydf.replace({',': ''}, regex=True).astype('float')
    summarydf = meanNorm(summarydf)
    summarydf = summarydf.fillna(value=0)
    '''
    detailsdf.index = idx
    detailsdf = detailsdf.fillna(value=0).replace({',': ''}, regex=True).astype('float')
    detailsdf1 = meanNorm(detailsdf)
    detailsdf1 = detailsdf1.fillna(value=0).transpose()
    '''detailsdf2 = percChange(detailsdf).fillna(value=0).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').transpose()'''
    #print(summarydf); print(detailsdf)
    #summarydf.plot(); detailsdf.plot(); plt.show()
    #kmeansWithpca(detailsdf1, plot=True)
    #kmeansWithpca(detailsdf2, plot=True) ##Failed with extreme data!!! fuck!!! Fuck? drop all 1 and 0?
    #correLation(detailsdf)
    #correLation(detailsdf1)
    try:
      hierarchicalClustering(detailsdf) ## too many data and cant plot... fuck... this not ok! why?
    except:
      print('Error')
      pass
    try:
      hierarchicalClustering(detailsdf1) ## this not ok! why?
	except:
      print('Error')
      pass

if __name__ == '__main__':
  pass
