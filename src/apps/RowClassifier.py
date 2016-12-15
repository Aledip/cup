# -*- coding: utf-8 -*-
'''
Created on 07 ott 2016

@author: Alejandro
'''

from builtins import IOError
import traceback

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier

from apps.ModelsManager import Classifier
from utils.DataUtils import DataUtils
from utils.TextUtils import TextUtils


fname = '/home/alejandro/Documenti/training_set.csv'  # '/home/alejandro/Documenti/Xand3cat.csv' #/home/alejandro/Documenti/VISIONATI.csv' 
del_char = "|"
limit = 1000000
fnameNonVisionati = '/home/alejandro/Documenti/NON_VISIONATI.csv'
campione = 169750
col_desc = 0
col_cat = ["AREA_INTERVENTO"]#["AREA_INTERVENTO", 'SETTORE_INTERVENTO', 'SOTTOSETTORE_INTERVENTO', 'CATEGORIA_INTERVENTO']
n_gram = (1, 2)  # None

data_utils = DataUtils()
text_utils = TextUtils()

models = []

models.append(PassiveAggressiveClassifier(n_iter=10, n_jobs=-1))
#models.append(RandomForestClassifier(n_jobs=-1))

#===================================================================
        # if(hash_features != None):
        #     self.count_vect = HashingVectorizer(non_negative=True, n_features=self.hash_features, ngram_range=self.n_gram)
        #     self.count_vect_cat = HashingVectorizer(non_negative=True, n_features=self.hash_features, ngram_range=self.n_gram)
        # else:
        #     self.count_vect = CountVectorizer(ngram_range=self.n_gram)
        #     self.count_vect_cat = CountVectorizer(ngram_range=self.n_gram)
        #===================================================================
         
        #self.tfidf_transformer = TfidfTransformer(use_idf=False)
        
        # models.append(MultinomialNB(alpha=0.001))
        #models.append(PassiveAggressiveClassifier(n_iter=10, n_jobs=-1))
        # models.append(SGDClassifier(loss='perceptron', alpha=0.001, n_iter=100, n_jobs=-1))
        # models.append(Perceptron(alpha=0.001, n_iter=100, n_jobs=-1, random_state=1))
        #models.append(RandomForestClassifier(n_jobs=-1))

        # self.models['AB'] = AdaBoostClassifier(base_estimator=self.models['NB'], n_estimators=100)
        # models.append( VotingClassifier(estimators=[('NB', models[0]), ('RF', models[4]), ('LR', models[5])], voting='soft', weights=[2,2,1]) )
        # models.append(GridSearchCV(estimator=models['NB'], param_grid=params, cv=5))



#classifier = Classifier(HashingVectorizer(non_negative=True,n_features = 15000, ngram_range=n_gram),f_select=10000)
classifier = Classifier(CountVectorizer(ngram_range=n_gram),models,f_select=100000,CV = 10) #fs_features = 95000
    
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
    
    #cross_validation(classifier, 5, X,cat_targets["AREA_INTERVENTO"] )
    #for cat in col_cat:
    #    cross_validation(classifier, 5, X, cat_targets[cat])
    
    
    #traceback.print_exc()
    print('models training..')
    classifier.train(X, cat_targets["AREA_INTERVENTO"])
    classifier.save()
    
s = input('>>')

#for label in col_cat:
      
#    print('\nfor ' + label + ' classified like:\n')

for k, v in classifier.classify([s]).items():
    print(k + " --> " + v)
    





