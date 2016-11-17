# -*- coding: utf-8 -*-
'''
Created on 07 ott 2016

@author: Alejandro
'''

from builtins import IOError
import traceback

from apps.Classifier_oop import Classifier
from utils.DataUtils_oop import DataUtils
from utils.TextUtils_oop import TextUtils


fname = '/home/alejandro/Documenti/training_set.csv'#'/home/alejandro/Documenti/Xand3cat.csv' #/home/alejandro/Documenti/VISIONATI.csv' 
del_char = "|"
limit = 1000000
fnameNonVisionati = '/home/alejandro/Documenti/NON_VISIONATI.csv'
campione = 169750
col_desc = 0
col_cat = ["AREA_INTERVENTO",'SETTORE_INTERVENTO','SOTTOSETTORE_INTERVENTO','CATEGORIA_INTERVENTO']
n_gram = (1, 2) #None

data_utils = DataUtils()
text_utils = TextUtils()

print('loading data..')
df = data_utils.csvReader(fname, limit, del_char)
print('Done')

print('sampling data..')
lista = data_utils.gen_XandY(df, campione, col_desc, col_cat)
print('Done')

X = lista[0]
targets = lista[1]

classifier = Classifier()
    
try:
    classifier.load()
    raise OSError
except (OSError, IOError) as e:
    #traceback.print_exc()
    print('models training..')
    classifier.train(X,targets,n_gram)
    classifier.save()
    
s = input('>>')

for label in targets:
      
    print('\nfor '+label+' classified like:\n')
    
    for k,v in classifier.classify([s], label).items():
        print(k+" --> "+v)
    





