'''
Created on Feb 6, 2016

@author: billperlman
'''
from pandas.io.parsers import read_csv
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
                          adjopens,
                          adjhighs,
                          adjlows,
                          adjcls]).T
    ret.columns = ['Contract','Date','Open','High','Low','Close','Volume',
                   'AdjOpen','AdjHigh','AdjLow','AdjClose']
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
    dates = data['Date'].apply(lambda yyyyMmDd:getYyyyMmDdFromYahooDate(yyyyMmDd)*100*100*100)
    df = commonFormat([None]*len(data), dates, data['Open'], data['High'], 
                data['Low'], data['Close'], data['Volume'], data['Adj Close'])
    return df

def dsInsert(ds,index, value):
    ''' insert a value into a Pandas Series '''
    temp = ds.tolist()
    temp.insert(index,value)
    return pd.Series(temp)

