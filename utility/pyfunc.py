'''
Created on Feb 6, 2016

@author: billperlman
'''
from pandas.io.parsers import read_csv
#from __builtin__ import None
'''

'''
import os
import re
import time 
import numpy as np
import datetime
import pandas as pd
pd.set_option('precision', 15)
from pandas import  DataFrame
from pandas.io.data import DataReader
import csv

monthVector =  ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def pasteListElements(l,sep):
    """ create a new string cancatenating the elemnts of l and separating them with sep 
        args:
            l - list
            sep - a string (like "," or "+") that will separate the list items
        
        Example:
            $ print(pasteListElements(['a','b','c'],","))
            $ 'a,b,c'
    """    
    ret = sep.join(l)
    return ret 

def lst(regexString):
    """ find all occurrences of regexString in the global namesapce
            this is very useful for finding functions that you have defined
            in your interpreter
        Example:
            $ print(lst('paste'))
            $ ['pasteListElements']
    """
    
    def _f(x):
        s = re.search(regexString,x)
        if s:
            return s.group(0)
        else:
            return None
        
    d = sorted(globals())
    ret = filter(_f, d)
    return ret 


def getYyyyMmDdFromYahooDate(sysDate=time.strftime("%Y-%m-%d")):
    ''' convert a date that comes from yahoo to a yyyyMmDd number 
        Example:
            $ print(getYyyyMmDdFromYahooDate('2015-02-24'))
            $ 20150224
    '''
    y = int(sysDate[0:4])
    mo = int(sysDate[5:7])
    day = int(sysDate[8:10])
    return y*100*100+mo*100+day

def getYyyyMmDdFromBarChartDate(sysDate=time.strftime("%m/%d/%Y")):
    ''' convert a date that comes from yahoo to a yyyyMmDd number 
        Example:
            $ print(getYyyyMmDdFromBarChartDate('02/24/2015'))
            $ 20150224
    '''
    y = int(sysDate[6:10])
    mo = int(sysDate[0:2])
    day = int(sysDate[3:5])
    return y*100*100+mo*100+day



def getTimeNumFromPosixDate(datetimeObj=None):
    ''' return yyyyMmDdHhMmSs time number from system date
        args: datetimeObj - a datetime object or None
        
        Example:
            $ print(getTimeNumFromPosixDate())
            $ print(getTimeNumFromPosixDate(datetime.datetime(2013, 11, 19, 14, 33, 59)))
            $ 20160206000000
            $ 20131119000000
    '''
    if datetimeObj:
        return getYyyyMmDdFromYahooDate(str(datetimeObj)) * 100*100*100
    else:
        return getYyyyMmDdFromYahooDate(str(datetime.datetime.now())) * 100*100*100
        

    
def commonFormat(symbols,yyyyMmDdHhMmSs,opens,highs,lows,closes,volumes,adjcloses=[]):
    ''' create a DataFrame where the columns are these vectors 
        args:
            symbols - symbol names
            yyyyMmDdHhMmSs - timenum int
            opens, highs, lows, closes, volumes - self explanatory
    '''
    percadjs = np.array([None]*len(yyyyMmDdHhMmSs))
    adjopens = percadjs
    adjhighs = percadjs
    adjlows = percadjs
    adjcls = percadjs
    
    if len(adjcloses)>0:
        adjcls = adjcloses
        percadjs = np.array(closes)/np.array(adjcloses)
        adjopens = np.array(opens)*np.array(percadjs)
        adjhighs = np.array(highs)*np.array(percadjs)
        adjlows = np.array(lows)*np.array(percadjs)
    
    
    ret = DataFrame(data=[np.array(symbols),
                          np.array(yyyyMmDdHhMmSs),
                          np.array(opens),
                          np.array(highs),
                          np.array(lows),
                          np.array(closes),
                          np.array(volumes),
                          adjcls,
                          adjopens,
                          adjhighs,
                          adjlows,
                          adjcls]).T
    ret.columns = ['Contract','Date','Open','High','Low','Close','Volume',
                   'Adjusted','AdjOpen','AdjHigh','AdjLow','AdjClose']
    return ret

todayYyyyMmDd = getYyyyMmDdFromYahooDate()

def readData(url,hasHeader=True):
    ''' csv read using pandas 
        Example:
        $ k =  readData('AIG.csv')
        $ print type(k)
        $ ibmyahooUrl = 'http://real-chart.finance.yahoo.com/table.csv?s=IBM&d=1&e=7&f=2016&g=d&a=0&b=2&c=1962&ignore=.csv';
        $ print readData(ibmyahooUrl)[0:5]

    '''
    ret = read_csv(url,header=0 if hasHeader else None)
    
    return ret 

def readBarChartCsv(filename='ibmBarchart.csv',hasHeader=False):
    ''' read csv file formated like the output of the BarChart csv downloads
        Example:
            $ d = readBarChartCsv('ibmBarchart.csv')
            $ print d
    '''
    data = readData(filename,hasHeader=hasHeader)
    # convert date
    data = data.sort(1)
    dates = data[1].apply(lambda mmDdYyyy:getYyyyMmDdFromBarChartDate(mmDdYyyy)*100*100*100)
    df = commonFormat(data[0], dates, data[2], data[3], 
                      data[4], data[5], data[6])
    return(df)


def readQuandlData(secname='CLZ2017',quandlPrefix='https://www.quandl.com/api/v3/datasets',
            exchange='CME',suffix='csv'):
    ''' read data (especially commodity data) from Quandl
        Example:
            $ print readQuandlData()[0:5] # get default values fro CLZ2017
    '''
    filename = quandlPrefix + '/' + exchange + '/' + secname + '.' + suffix
    data = readData(filename)
    data = data.sort('Date')
    dates = data['Date'].apply(lambda yyyyMmDd:getYyyyMmDdFromYahooDate(yyyyMmDd)*100*100*100)
    df = commonFormat([None]*len(data), dates, data['Open'], data['High'], 
                data['Low'], data['Settle'], data['Volume'])
    return df

def readYahoo(secname='SPY',begYyyyMmDd=20060101,endYyyyMmDd=getTimeNumFromPosixDate()):
    ''' read from yahoo 
        Example:
            $ print readYahoo()[0:5]
    '''
    begy = str(begYyyyMmDd)[0:4]
    begm = str(begYyyyMmDd)[4:6]
    begd = str(begYyyyMmDd)[6:8]
    endy = str(endYyyyMmDd)[0:4]
    endm = str(endYyyyMmDd)[4:6]
    endd = str(endYyyyMmDd)[6:8]
    httpstring = "http://real-chart.finance.yahoo.com/table.csv?s=STOCK&d=ENDM&e=ENDD&f=ENDY&g=d&a=BEGM&b=BEGD&c=BEGY&ignore=.csv"
    httpstring = httpstring.replace("BEGY",begy)       
    httpstring = httpstring.replace("BEGM",begm)    
    httpstring = httpstring.replace("BEGD",begd)
    httpstring = httpstring.replace("ENDY",endy)
    httpstring = httpstring.replace("ENDM",endm)
    httpstring = httpstring.replace("ENDD",endd)
    httpstring = httpstring.replace("STOCK",secname)
    
    
    data = readData(httpstring)
    data = data.sort('Date')
    dates = data['Date'].apply(lambda yyyyMmDd:getYyyyMmDdFromYahooDate(yyyyMmDd)*100*100*100)
    df = commonFormat([None]*len(data), dates, data['Open'], data['High'], 
                data['Low'], data['Close'], data['Volume'], data['Adj Close'])
    return df

def dsInsert(ds,index, value):
    ''' insert a value into a Pandas Series '''
    temp = ds.tolist()
    temp.insert(index,value)
    return pd.Series(temp)


def asb(np1,np2, boolarray):
    ''' emulate the r construct of x[b] = y[b]
        Example:
            $ x = np.array([0]*10)
            $ y = np.array(range(10))
            $ b = np.array([True,False]*5)
            $ print asb(x,y,b)
            $ [0, 0, 2, 0, 4, 0, 6, 0, 8, 0]
            $ b  = np.array(y%2!=0)
            $ print asb(x,y,b)
            $ [0, 1, 0, 3, 0, 5, 0, 7, 0, 9]
            $ x = [0]*10
            $ y = range(10)
            b = np.ndarray.tolist(b)
            $ print asb(x,y,b)
            $ [0, 1, 0, 3, 0, 5, 0, 7, 0, 9]
    '''
    def f(x,y,b):
        if b:
            return y
        else:
            return x
        
    m = map(lambda x,y,b:f(x,y,b),np1,np2,boolarray)
    return np.array(m)

def pseudoStockFromReturns(dfWithDateAndReturnsCols):
    ''' create a psuedo stock from returns
        args:
            dfWithDateAndReturnsCols - a Dataframe with 2 cols - Date and ret
    '''
    pseudostk = np.cumprod(dfWithDateAndReturnsCols[:,1]+1)
    df = DataFrame({'Date':dfWithDateAndReturnsCols[:,0],'Adjusted':pseudostk,'AdjPrev':[1,pseudostk[0:(len(pseudostk)-1)]]})
    return df

'''
returnsPerformance <- function(returnsDf,printit=TRUE){
    psu <- pseudoStockFromReturns(returnsDf)
    return(stockPerformance(dataForStock=psu,printit=printit))
'''
def stockPerformance(symbol='SPY',
                    begYyyyMmDd=19990101,
                    endYyyyMmDd=getYyyyMmDdFromYahooDate(),
                    daysOfSd=50,
                    printit=True,
                    dataForStock=None,
                    entryCol='AdjPrevClose',
                    exitCol='Adjusted'):
    stockData=dataForStock
    if stockData==None:
        try:
            stockData = readYahoo(symbol, begYyyyMmDd, endYyyyMmDd)
        except Exception, e:
            print 'error getting data for ' + symbol + 'err = ' + e
    try:
        ''' need to fake out older versions of Pandas to check for null'''
        b = stockData==None
        if b:
            print 'error getting data for ' + symbol 
    except:
        ''' do nothing and fall through '''
        b = None
    
    stockData['AdjPrevClose'] = dsInsert(stockData['Adjusted'][0:(len(stockData)-1)],0,None)
    stockData['PrevClose'] = dsInsert(stockData['Close'][0:(len(stockData)-1)],0,None)
    stockData = stockData[1:]
    stockData.index = range(len(stockData))
    firstYyyyMmDd = stockData.ix[0,'Date']/(100*100*100)
    firstYear = int(str(firstYyyyMmDd)[0:4])
    firstMonth = int(str(firstYyyyMmDd)[4:6])
    firstDay = int(str(firstYyyyMmDd)[6:8])
    retdat = stockData.ix[:,exitCol] / stockData.ix[:,entryCol] - 1
    m = retdat.mean()
    s = retdat.std()
    sh = m/s * 252**.5
    cm = np.cumprod(retdat)
    listMeans = pd.rolling_mean(retdat,daysOfSd)
    listSds = pd.rolling_std(retdat,daysOfSd)
    listShs = listMeans/listSds * 252**.5
    stockData['rollsh'] = listShs
    stockData['cumret'] = cm
    
    return {'mean':m,'sd':s,'sharpe':sh,'stockData':stockData}    
        

def returnsPerformance(returnsDf,printit=True):
    psu = pseudoStockFromReturns(returnsDf)
    return stockPerformance(dataForStock=psu,printit=printit)

s = stockPerformance()
print s['mean']
print s['sd']
print s['sharpe']
# print(readBarChartCsv()[0:20])
# print(readYahoo()[0:20])
# print(readQuandlData()[0:20])

