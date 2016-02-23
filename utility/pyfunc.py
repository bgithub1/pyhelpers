'''
Created on Feb 6, 2016

@author: billperlman
'''
from pandas.io.parsers import read_csv
import os
import re
import time 
import numpy as np
import datetime
import pandas as pd
pd.set_option('precision', 15)
from pandas import  DataFrame, Series
from pandas.io.data import DataReader
import csv
import matplotlib.pyplot as plt
from peakdetect import peakdetect



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
    pseudostk = np.cumprod(dfWithDateAndReturnsCols.iloc[:,1]+1)
    df = DataFrame({'Date':dfWithDateAndReturnsCols.iloc[:,0],'Adjusted':pseudostk,'AdjPrev':dsInsert(pseudostk[0:(len(pseudostk)-1)],1,0)})
    return df

def rbind(df,newrec):
    ''' r-like append of row to DataFrame'''
    newdf = DataFrame(newrec).T
    newdf.columns = df.columns
    ret = df.append(newdf)
    ret.index = range(len(ret))
    return ret


    

def names(obj):
    ''' like names(obj) in r '''
    return [key for key, val in obj.iteritems()]

def peakToTroughs(dailyret,dates):
    '''
    '''
    ''' get cummulative percent changes'''
    drs = Series(dailyret)
    soc1dr = drs+1
    soc1cumdr = soc1dr.cumprod()
    localPeaksPairs = peakdetect(y_axis=soc1cumdr,lookahead=1)[0]
    indexOfLocalPeaks  = np.empty(len(localPeaksPairs));
    for i in range(len(indexOfLocalPeaks)):
        indexOfLocalPeaks[i] = localPeaksPairs[i][0]
    # data frame with 2 columns, where column 1 is a peak, and column 2 is the next peak that follows it
    dd = DataFrame({'a':indexOfLocalPeaks[0:(len(indexOfLocalPeaks)-1)],'b':indexOfLocalPeaks[1:len(indexOfLocalPeaks)]})
    # add one more row to dd to represent the last peak and last row of soc1cumdr, so
    #   that you calculate the last possible trough, if it there was one between the last peak and the last day
    #   of data
    lastDdValue = dd.iloc[len(dd)-1,1]
    lastValueInData = len(soc1cumdr)-1
    dd = rbind(dd,[lastDdValue,lastValueInData])
    def minBetween2Peaks(x):
        lowindex = int(x[0])
        highindex = int(x[1])
        minval = min(soc1cumdr[lowindex:(highindex+1)])
        return minval
    localMins = dd.apply(minBetween2Peaks,1)
    localMins.index = range(len(localMins))
    localPeaks = soc1cumdr[indexOfLocalPeaks.astype(int)]
    localPeaks.index = range(len(localPeaks))
    diffs = (localMins - localPeaks)/localPeaks
    
    # get indices of localMins in soc1cumdr so that you can get their dates
    def ff(x):
        ''' this function gets the index of soc1cumdr whose value = x'''
        r = soc1cumdr[soc1cumdr==x].index[0]
        return r
    indexOfLocalMins = map(ff,localMins)
    datesOfLocalMins = Series(dates)[indexOfLocalMins]
    datesOfLocalMins.index = range(len(datesOfLocalMins))
    # calculate peak to end of data
    def minBetweenPeakAndEnd(x):
        arr = soc1cumdr.iloc[x[0]:len(soc1cumdr)]
        return min(arr)
    absMinsToEnd = dd.apply(minBetweenPeakAndEnd,1)
    absMinsToEnd.index = range(len(absMinsToEnd))
    diffsToEnd = (absMinsToEnd - localPeaks)/localPeaks
    ret =  DataFrame({'Date':datesOfLocalMins,'Peak':localPeaks,'Valley':localMins,'Diff':diffs,'DiffToEnd':diffsToEnd})

    return ret
    
      
'''
#'    @param soc: output from strat_openClose
peakToTroughs <- function(dailyret,dates){
    soc1dr <- dailyret
    soc1cumdr <- cumprod(soc1dr+1)
    indexOfLocalPeaks <- findPeaks(soc1cumdr)-1 # find peaks
    # data frame with 2 columns, where column 1 is a peak, and column 2 is the next peak that follows it
    dd <- data.frame(indexOfLocalPeaks[1:(length(indexOfLocalPeaks)-1)],indexOfLocalPeaks[2:length(indexOfLocalPeaks)])
    # add one more row to dd to represent the last peak and last row of soc1cumdr, so
    #   that you calculate the last possible trough, if it there was one between the last peak and the last day
    #   of data
    dd <- rbind(dd,c(dd[nrow(dd),2],nrow(soc1cumdr)))
    minBetween2Peaks <- function(x){return(min(soc1cumdr[x[1]:x[2]]))}
    localMins <- apply(dd,1,minBetween2Peaks)
    localPeaks <- soc1cumdr[indexOfLocalPeaks]
    diffs <- (localMins - localPeaks)/localPeaks
    # get indices of localMins in soc1cumdr so that you can get their dates
    indexOfLocalMins <- sapply(localMins,function(x){which(soc1cumdr==x)[1]})
    
    datesOfLocalMins <- dates[indexOfLocalMins]
    # calculate peak to end of data
    minBetweenPeakAndEnd <- function(x){return(min(soc1cumdr[x[1]:length(soc1cumdr)]))}
    absMinsToEnd <- apply(dd,1,minBetweenPeakAndEnd)
    diffsToEnd <- (absMinsToEnd - localPeaks)/localPeaks


    ret <- data.frame(cbind(datesOfLocalMins,localPeaks,localMins,diffs,diffsToEnd))
    colnames(ret) <- c('Date','Peak','Valley','Diff','DiffToEnd')
    return(ret)
}
'''


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
                    exitCol='Adjusted',
                    block=False):
    stockData=dataForStock
    if stockData is None:
        stockData = readYahoo(symbol, begYyyyMmDd, endYyyyMmDd)

    stockData[entryCol] = dsInsert(stockData[exitCol][0:(len(stockData)-1)],0,None)
    stockData['PrevClose'] = dsInsert(stockData[exitCol][0:(len(stockData)-1)],0,None)
    stockData = stockData[1:]
    stockData.index = range(len(stockData))
    firstYyyyMmDd = stockData.loc[0,'Date']/(100*100*100)
    firstYear = int(str(firstYyyyMmDd)[0:4])
    firstMonth = int(str(firstYyyyMmDd)[4:6])
    firstDay = int(str(firstYyyyMmDd)[6:8])
    retdat = stockData.ix[:,exitCol] / stockData.ix[:,entryCol] - 1
    m = retdat.mean()
    s = retdat.std()
    sh = m/s * 252**.5
    cm = np.cumprod(retdat+1)-1
    listMeans = pd.rolling_mean(retdat,daysOfSd)
    listSds = pd.rolling_std(retdat,daysOfSd)
    listShs = listMeans/listSds * 252**.5
    stockData['rollsh'] = listShs
    stockData['cumret'] = cm
    pt = peakToTroughs(retdat,stockData['Date'])
    shavg = stockData.rollsh[(daysOfSd+1):len(stockData)].mean()
    ret = {'mean':m,'sd':s,'sharpe':sh,'sharpeAvg':shavg,'peakToTrough':pt,'stockData':stockData,'retdat':retdat,'firstYyyyMmDd':firstYyyyMmDd}
    if printit:
        printStockPerformance(ret,block=block)
    return ret    
    '''
    
    '''    
def makeReturns(arr):
    ''' '''
    

def returnsPerformance(returnsDf,printit=True):
    psu = pseudoStockFromReturns(returnsDf)
    return stockPerformance(dataForStock=psu,printit=printit)

def printStockPerformance(stockPerformanceResults,block=False):
    '''
    s <- stockPerformanceResults
    par(mfrow=c(2,2))
    plotDf(s$stockData,priceCol='cumret',ylow=NULL,nameOfDf='cumret')
    plotDf(s$stockData,priceCol='rollsh',ylow=NULL,nameOfDf='rolling sharpe')
    hist(s$retdat,breaks=50)
    hist(s$stockData$rollsh,breaks=50)
    print(paste('as:',s$firstYyyyMmDd,' mean=',round(s$mean,4),' sd=',round(s$sd,4),' sharpe=',round(s$sharpe,4),' avg sharpe=',round(s$sharpeAvg,4),sep=""))
    m <- round(min(s$peakToTrough$DiffToEnd),4)
    print(paste('worst peak to trough =',m))
    m <- round(median(s$peakToTrough$DiffToEnd),4)
    print(paste('median peak to trough =',m))
    print('last 10 days of rolling sharpe')
    print(s$stockData[(nrow(s$stockData)-10):nrow(s$stockData),c('Date','rollsh')])
    
    '''
    plt.clf()
    s = stockPerformanceResults
    plotDf(s['stockData'],priceCol='cumret',subplotArray=[2,2,1],showit=False)
    plotDf(s['stockData'],priceCol='rollsh',subplotArray=[2,2,2],showit=False)
    plt.subplot(2,2,3)
    s['retdat'].hist(bins=50)
    plt.subplot(2,2,4)
    s['stockData']['rollsh'].hist(bins=50)
    plt.tight_layout()
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("800x900+50+50")
    print('as:' + str(s['firstYyyyMmDd']) + ' mean=' + str(round(s['mean'],4)) + 
                      ' sd=' + str(round(s['sd'],4)) + ' sharpe='  + str(round(s['sharpe'],4)) + 
                      ' avg sharpe=' + str(round(s['sharpeAvg'],4)) )
    ptp = s['peakToTrough']['DiffToEnd']
    min1 = round(ptp.min(),4)
    print('worst peak to trough =' + str(min1))
    med1 = round(ptp.median(),4)
    print('median peak to trough =' + str(med1))
    print('last 10 days of rolling sharpe')
    print(s['stockData'][(len(s['stockData'])-10):len(s['stockData'])][['Date','rollsh']])
    plt.show(block=block)  
    
    
    
def plotDf(df,dateCol="Date",priceCol="Adjusted",subplotArray=None,showit=True):
    ''' '''
    date = df[dateCol]
    x = range(len(date))
    p = df[priceCol]
    r = range(0,len(date),len(date)/20)
    dtc = map(lambda a:'%d' % a,date[r])
    if subplotArray:
        plt.subplot(subplotArray[0],subplotArray[1],subplotArray[2])
    plt.plot(x,p);plt.xticks(r,dtc,rotation=45,fontsize=9);plt.gcf().subplots_adjust(bottom=.18)
    if showit:
        plt.show()
    
def testsubplot():
    x1 = np.linspace(0.0, 5.0)
    x2 = np.linspace(0.0, 2.0)
    y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    y2 = np.cos(2 * np.pi * x2)
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'ko-')
    plt.title('A tale of 2 subplots')
    plt.ylabel('Damped oscillation')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, 'r.-')
    plt.xlabel('time (s)')
    plt.ylabel('Undamped')
    plt.show()

block=False
s = stockPerformance(begYyyyMmDd=20060101,block=block)
sr = s['retdat']
stkd = s['stockData']
dt = stkd['Date']
#ptk = peakToTroughs(sr,dt)

# print(readBarChartCsv()[0:20])
# print(readYahoo()[0:20])
# print(readQuandlData()[0:20])

