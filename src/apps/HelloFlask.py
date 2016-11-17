# -*- coding: utf-8 -*-
'''
Created on 17 ott 2016

@author: alejandro
'''
import codecs
import itertools
import json

from flask import Flask
from flask.globals import request
from flask.wrappers import Response

app = Flask(__name__)

def checkjson(request):
    return request.headers.get("Content-Type").startswith("application/json")

class ServiceError:
    def __init__(self, msg, status=400):
        self.msg = msg
        self.status = status
       
def tojson(f):
    def fun(*args, **kwargs):
        if request.method in ["POST", "PUT"] and not checkjson(request):
            err = ServiceError("Please set header 'Content-Type=application/json' ")
            return Response(err.msg, err.status)
        result = f(*args, **kwargs)
        status = 200
        if isinstance(result, ServiceError):
            result, status = result.msg, result.status
        if isinstance(result, Response):
            return result
        dump = json.dumps(result, default=lambda o: o.__dict__)
            
        return Response(dump, status=status, mimetype='application/json')
    fun.__name__ = f.__name__
    fun.__doc__ = f.__doc__
    return fun

@app.route("/opencup/slices/<int:start>/<int:end>")
@tojson
def slices(start, end):
    data = codecs.open('/home/alejandro/Documenti/Xand3cat.csv', "r", "ISO-8859-1")
    ret = []
    header = data.__next__().split('|')
    for line in itertools.islice(data, start, end):
        temp = line.strip().split("|")
        ret.append(dict(zip(header, temp)))    
    return ret

#===============================================================================
# @app.route("/opencup/classify/<str:input>")
# @tojson
# def classify(input):
#     
#     targets = ["AREA_INTERVENTO", 'SETTORE_INTERVENTO', 'SOTTOSETTORE_INTERVENTO', 'CATEGORIA_INTERVENTO']
#     classifier = Classifier()
#     classifier.load()
#     ris = dict()
#     for label in targets:
#       
#         for k, v in classifier.classify([input], label).items():
#             print( k + " --> " + v )
#             ris[k] = v
#             
#     return [ris]
#===============================================================================

if __name__ == "__main__":
    app.run(debug=True)
