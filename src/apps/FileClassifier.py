'''
Created on 22 nov 2016

@author: alejandro
'''

from builtins import IOError
import string
import time
import traceback

import pandas
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier

from apps.ModelsManager import Classifier
from utils.DataUtils import DataUtils
from utils.TextUtils import TextUtils


fname = '/home/alejandro/Scrivania/TOTALE.csv'  # '/home/alejandro/Documenti/Xand3cat.csv' #/home/alejandro/Documenti/VISIONATI.csv' 
#fname = '/home/alejandro/Scrivania/test.csv'  # '/home/alejandro/Documenti/Xand3cat.csv' #/home/alejandro/Documenti/VISIONATI.csv' 

del_char = '|'
limit = 1000000
fnameNonVisionati = '/home/alejandro/Documenti/NON_VISIONATI.csv'
campione = 100 #169750
col_desc = 0
col_cat = ["AREA_INTERVENTO", 'SETTORE_INTERVENTO', 'SOTTOSETTORE_INTERVENTO', 'CATEGORIA_INTERVENTO']
n_gram = (1, 2)  # None

data_utils = DataUtils()
text_utils = TextUtils()

models = []

models.append(PassiveAggressiveClassifier(n_iter=10, n_jobs=-1))

classifier = Classifier(HashingVectorizer(non_negative=True,n_features = 5000, ngram_range=n_gram),models)
    
#-------------------------------------------------------------------------- try:
    #--------------------------------------------------------- classifier.load()
#----------------------------------------------- except (OSError, IOError) as e:
#------------------------------------------------------------------------------ 
    #--------------------------------------------------- print('loading data..')
    # df = data_utils.csvReader('/home/alejandro/Scrivania/test.csv', limit, del_char)
    #------------------------------------------------------------- print('Done')
#------------------------------------------------------------------------------ 
    #-------------------------------------------------- print('sampling data..')
    #---- X, cat_targets = data_utils.gen_XandY(df, campione, col_desc, col_cat)
    #------------------------------------------------------------- print('Done')
#------------------------------------------------------------------------------ 
    #---------------------------------------------------- #traceback.print_exc()
    #------------------------------------------------ print('models training..')
    #------------------------------------------ classifier.train(X, cat_targets)
    #--------------------------------------------------------- classifier.save()


t = time.time() 
s = ''
c = 0

df = data_utils.csvReader(fname, 2000000, "|")
print()
dfn = pandas.DataFrame()
df = df[df["CONTROLLO_QUALITA"] == "Il corredo informativo di questo CUP è stato visionato, almeno parzialmente"]
print(len(df))
print(type(df.iloc[:,17]),df.iloc[:,17])

print(dfn)
dfn.to_csv("/home/alejandro/Scrivania/prova2.csv")
 
    
clist = df.columns
print(list(clist))


#-- with open("/home/alejandro/Scrivania/TOTALE.csv",'r',encoding='latin') as f:
    #---------------------------------------------------------- header = next(f)
    #------------------------------------------------------------- print(header)
#------------------------------------------------------------------------------ 
    #------------------------------------------------------------- for row in f:
#------------------------------------------------------------------------------ 
        #--------------------------------------------- rowsplit = row.split("|")
        #---------------------------------------------------- desc = rowsplit[1]
#------------------------------------------------------------------------------ 
        #-------------------------------------------------------------- ris = []
        #--------------------------------------------------- for col in col_cat:
            # ris.append(classifier.classify(desc, col)['PassiveAggressiveClassifier'])
#------------------------------------------------------------------------------ 
        #----------------------------- s +=row[:-1] + "|" + ",".join(ris) + "\n"
        #--------------------------------- print(row[:-1] + "|" + ",".join(ris))
        #------------------------------------------------------------------ c+=1
        #-------------------------------------------------------------- if c==3:
            #------------------------------------------------------------- break
#------------------------------------------------------------------------------ 
    #---------- with open("/home/alejandro/Scrivania/TOTALE_CLF.txt",'w') as f2:
        #----------------------------------------------------------- f2.write(s)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------ print(str(round(time.time() - t, 3)) + "s for classify TOTAL.csv")
        


