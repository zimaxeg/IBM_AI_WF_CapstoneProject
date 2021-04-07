#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time,os,re,csv,sys,uuid,joblib
from datetime import date

if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")


def update_train_log(tag,data_shape,eval_test,runtime,MODEL_VERSION,MODEL_VERSION_NOTE,test=False):
    """
    update train log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs","{}-train-test.log".format(tag))
    else:
        logfile = os.path.join("logs","sl-{}-train-{}-{}.log".format(tag,today.year, today.month))
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','x_shape','eval_test','model_version',
              'model_version_note','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),data_shape,eval_test,
                            MODEL_VERSION,MODEL_VERSION_NOTE,runtime])
        writer.writerow(to_write)

def update_predict_log(country, y_pred,y_proba,query,runtime,MODEL_VERSION,test=False):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs","predict-test.log")
    else:
        logfile = os.path.join("logs","predict-{}-{}.log".format(today.year, today.month))
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','y_pred','y_proba','query','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),y_pred,y_proba,query,
                            MODEL_VERSION,runtime])
        writer.writerow(to_write)
 
#def log_load(tag,year,month,env,verbose=True):
def log_load(tag,env,env1,verbose=True):
    """
    load requested log file
    """
#    logfile = "{}-{}-{}-{}.log".format(env,tag,year,month)
    logfile = "{}-{}-{}.log".format(tag,env,env1)
    
    if verbose:
        print(logfile)
    return logfile
    