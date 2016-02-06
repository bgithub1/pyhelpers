'''
Created on Feb 6, 2016

@author: billperlman
'''
'''

'''
import re
import time 
import datetime



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


#int(time.strftime("%Y%m%d"))

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
        

