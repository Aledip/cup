'''
Created on 22 nov 2016

@author: alejandro
'''

from builtins import IOError
import string
import time
import traceback

from sklearn.feature_extraction.text import HashingVectorizer

from apps.ModelsManager import Classifier
from utils.DataUtils import DataUtils
from utils.TextUtils import TextUtils


fname = '/home/alejandro/Documenti/training_set.csv'  # '/home/alejandro/Documenti/Xand3cat.csv' #/home/alejandro/Documenti/VISIONATI.csv' 
del_char = "|"
limit = 1000000
fnameNonVisionati = '/home/alejandro/Documenti/NON_VISIONATI.csv'
campione = 100 #169750
col_desc = 0
col_cat = ["AREA_INTERVENTO", 'SETTORE_INTERVENTO', 'SOTTOSETTORE_INTERVENTO', 'CATEGORIA_INTERVENTO']
n_gram = (1, 2)  # None

data_utils = DataUtils()
text_utils = TextUtils()


classifier = Classifier(HashingVectorizer(non_negative=True,n_features = 5000, ngram_range=n_gram))
    
try:
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


t = time.time() 
s = ''

with open("/home/alejandro/Scrivania/TOTALE.csv",'r',encoding='latin') as f:
    for row in f:
        desc = row.split("|")[1]
        print(desc)
        ris = []
        for col in col_cat:
            ris.append(classifier.classify(desc, col)['PassiveAggressiveClassifier'])
        s +=row + "|" + " ".join(ris)
        
    with open("/home/alejandro/Scrivania/TOTALE_CLF.txt",'w') as f2:
        f2.write(s)
        
        
print(str(round(time.time() - t, 3)) + "s for classify TOTAL.csv")
        


