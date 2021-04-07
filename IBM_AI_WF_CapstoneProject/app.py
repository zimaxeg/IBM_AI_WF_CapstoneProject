
import argparse
from flask import Flask, jsonify, request
from flask import render_template
import joblib
import socket
import json
import numpy as np
import pandas as pd
import os

## import model specific functions and variables
#from modelling import *
from logger import *
from model import *
app = Flask(__name__)

@app.route("/")
def hello():
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

@app.route('/predict', methods=['GET','POST'])

def predict():
    """
    basic predict function for the API
    """
   # return(jsonify({'y_pred': 206, 'y_proba': 1}))
    ## input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])

    if 'query' not in request.json:
        print("ERROR API (predict): received request, but no 'query' found within",request,"req-json",request.json)
        return jsonify([])

    if 'type' not in request.json:
        print("WARNING API (predict): received request, but no 'type' was found assuming 'numpy'")
        query_type = 'numpy'

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True

    ## extract the query
    query = request.json['query']
    country=query['country']
    year=query['year']
    month=query['month']
    day=query['day']
        
    if request.json['type'] == 'dict':
        pass
    else:
        print("ERROR API (predict): only dict data types have been implemented")
        return jsonify([])
        
    ## load model
    model = model_load()
    
    if not model:
        print("ERROR: model is not available")
        return jsonify([])

    #_result = model_predict(country,year,month,day,model,test=test)
    _result = model_predict(country,year,month,day)

    result = {}
    
    ## convert numpy objects to ensure they are serializable
    for key,item in _result.items():
        if isinstance(item,np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item
    print(result)
    return(jsonify(result))
    #return(jsonify({'y_pred': [183213.76200000005], 'y_proba': 1})) 


@app.route('/train', methods=['GET','POST'])
def train():
    """
    basic train function for the API

    the 'dev' give you the ability to toggle between a DEV version and a PROD verion of training
    """


    print("... training model")
    print(request.json)
    #model = model_train(dev=request.json['dev']=="True", verbose=verbose=="True")
    #model = model_train(dev=request.json['dev']=="True", verbose=request.json['verbose']=="True")
    model= model_train()
    print("... training complete")

    return(jsonify(True))

@app.route('/logging', methods=['GET','POST'])
def load_logs():
    """
    basic logging function for the API
    """

    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify(False)

    if 'env' not in request.json:
        print("ERROR API (log): received request, but no 'env' found within")
        return jsonify(False)
        
    if 'type' not in request.json:
        print("ERROR API (log): received request, but no 'type' found within")
        return jsonify(False)

    if 'tag' not in request.json:
        print("ERROR API (log): received request, but no 'tag' found within")
        return jsonify(False)
        
  
    print("... fetching logfile")
    logfile = log_load(env=request.json['env'],
                       tag=request.json['tag'],
                       env1=request.json['type'])
                    #   month=request.json['month'])
    
    result = {}
    result["logfile"]=logfile
    return(jsonify(result))

if __name__ == '__main__':

    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        #app.run(host='0.0.0.0', threaded=True ,port=8080)
        app.run(host='localhost', threaded=True ,port=8080)
