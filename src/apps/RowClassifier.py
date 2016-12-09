# -*- coding: utf-8 -*-
'''
Created on 07 ott 2016

@author: Alejandro
'''

from builtins import IOError
import traceback

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

from apps.ModelsManager import Classifier
from utils.DataUtils import DataUtils
from utils.TextUtils import TextUtils


fname = '/home/alejandro/Documenti/training_set.csv'  # '/home/alejandro/Documenti/Xand3cat.csv' #/home/alejandro/Documenti/VISIONATI.csv' 
del_char = "|"
limit = 1000000
fnameNonVisionati = '/home/alejandro/Documenti/NON_VISIONATI.csv'
campione = 100 #169750
col_desc = 0
col_cat = ["AREA_INTERVENTO"]#["AREA_INTERVENTO", 'SETTORE_INTERVENTO', 'SOTTOSETTORE_INTERVENTO', 'CATEGORIA_INTERVENTO']
n_gram = (1, 2)  # None

data_utils = DataUtils()
text_utils = TextUtils()



classifier = Classifier(HashingVectorizer(non_negative=True,n_features = 5000, ngram_range=n_gram))
    
try:
    raise IOError
    classifier.load()
except (OSError, IOError) as e:
    
    print('loading data..')
    df = data_utils.csvReader(fname, limit, del_char)
    print('Done')
    
    print('sampling data..')
    X, cat_targets = data_utils.gen_XandY(df, campione, col_desc, col_cat)
    print('Done')

    #traceback.print_exc()
    print('models training..')
    classifier.train(X, cat_targets)
    classifier.save()
    
s = input('>>')

for label in col_cat:
      
    print('\nfor ' + label + ' classified like:\n')
    
    for k, v in classifier.classify([s], label).items():
        print(k + " --> " + v)
    





