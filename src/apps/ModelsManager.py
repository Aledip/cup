# -*- coding: utf-8 -*-
'''
Created on 06 ott 2016

@author: Alejandro
'''
from _collections import defaultdict, OrderedDict
import os
import time

from scipy.sparse.construct import hstack
from sklearn import metrics
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.feature_selection.univariate_selection import chi2, SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.perceptron import Perceptron
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import numpy as np
from utils.DataUtils import DataUtils


class Classifier(object):
    
    
    def get_accuracies(self):
        return self.accuracies
    
    def get_n_gram(self):
        return self.n_gram
    
    def get_hash_features(self):
        return self.hash_features
    
    def set_models(self, models):
        self.models = models
    
    def get_models(self):
        return self.models
    
    def get_vectorizer(self):
        return self.vectorizer


    path = "/home/alejandro/models/"
    vectorizer = None
    count_vect = None
    count_vect_cat = None
    tfidf_transformer = None
    accuracies = []
    hash_features = None
    f_select = None
    CV = None
    models = None
    
    def __init__(self, vect, models, f_select=None, CV=None):
        self.vectorizer = vect
        self.models = models
        # self.fvectorizer = Vect
        self.f_select = f_select
        self.CV = CV
    
    
    
    def train(self, X, target):
        print('training..')
        
        if self.CV != None:
            self.cross_validation(self.CV, X, target)
        
        dict1 = OrderedDict()
        
        X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.70, random_state=42)
        
        params = {'alpha': [0.01, 0.001 , 0.0001], }
                    
        
        #===================================================================
        # if(hash_features != None):
        #     self.count_vect = HashingVectorizer(non_negative=True, n_features=self.hash_features, ngram_range=self.n_gram)
        #     self.count_vect_cat = HashingVectorizer(non_negative=True, n_features=self.hash_features, ngram_range=self.n_gram)
        # else:
        #     self.count_vect = CountVectorizer(ngram_range=self.n_gram)
        #     self.count_vect_cat = CountVectorizer(ngram_range=self.n_gram)
        #===================================================================
         
        # self.tfidf_transformer = TfidfTransformer(use_idf=False)
        
        # models.append(MultinomialNB(alpha=0.001))
        # models.append(PassiveAggressiveClassifier(n_iter=10, n_jobs=-1))
        # models.append(SGDClassifier(loss='perceptron', alpha=0.001, n_iter=100, n_jobs=-1))
        # models.append(Perceptron(alpha=0.001, n_iter=100, n_jobs=-1, random_state=1))
        # models.append(RandomForestClassifier(n_jobs=-1))

        # self.models['AB'] = AdaBoostClassifier(base_estimator=self.models['NB'], n_estimators=100)
        # models.append( VotingClassifier(estimators=[('NB', models[0]), ('RF', models[4]), ('LR', models[5])], voting='soft', weights=[2,2,1]) )
        # models.append(GridSearchCV(estimator=models['NB'], param_grid=params, cv=5))
        
        t_start_vect = time.time()   
        train_matrix = self.prepare_train_matrix(X_train, y_train)
        
        print(str(round(time.time() - t_start_vect, 3)) + "s for Vectorization with " + type(self.vectorizer).__name__ + "\n")
        
        for model in self.models:
            t_start = time.time()
            model.fit(train_matrix, y_train)
            print(str(round(time.time() - t_start, 3)) + "s for training " + type(model).__name__)
            
        
        accuracy = self.get_accuracy(X_test, y_test)

        for k, v in accuracy.items():
            print(str(k) + " : " + str(v))
            
        print("Done")
        print("\n#########################################################")
    
    
    def cross_validation(self, CV, X, y):
        
        print("cross validation..")
        X_t = self.get_vectorizer().fit_transform(X)
        if self.f_select != None:
            X_t = self.train_feature_selection(X_t, y)
        for model in self.get_models():
            scores = cross_val_score(model, X_t, y, cv=CV)
            # metrics.accuracy_score(X,y) 
            print(type(model).__name__ + " [Cross Validation] : " + str(scores.mean()))
            print(str(scores) + "\n")
        print("Done")
            
        
    
    def add_element(self, x_desc, y_label):
        
        for m in self.models:
            m.partial_fit(x_desc, y_label)
        
        
    def save(self):
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        name = "modelli_addestrati.pkl"
        print ("saving models...")
        t_start_vect = time.time()
        
        if not os.path.exists(self.path):
                os.makedirs(self.path)
                
        joblib.dump(self.models, self.path + name, compress=3)  # joblib.dump([self.labels, self.count_vect, self.count_vect_cat, self.tfidf_transformer], self.path + name, compress=3)
        print(str(round(time.time() - t_start_vect, 3)) + "s for save models")
        print("Done")
        
      
    def load(self):
        
        name = "modelli_addestrati.pkl"
        print("loading models...")
        t_start_vect = time.time()
        self.models = joblib.load(self.path + name)  # self.labels, self.count_vect, self.count_vect_cat, self.tfidf_transformer = joblib.load(self.path + name)
        print(str(round(time.time() - t_start_vect, 3)) + "s for load models")
        print("Done")
     
    def get_accuracy(self, X_test, y_test):
        accuracies = dict()
        test_matrix = self.prepare_input_matrix(X_test)
        
        for model in self.models:
            accuracies[type(model).__name__] = round(np.mean(model.predict(test_matrix) == y_test), 2)
        return accuracies
    
    def classify(self, X_test):
        clf_res = dict()
        test_matrix = self.prepare_input_matrix(X_test)
        for model in self.models:
            clf_res[type(model).__name__] = model.predict(test_matrix)[0]
        return clf_res
            
    
    def prepare_train_matrix(self, X_train, y_train):
        
        train_matrix = self.vectorizer.fit_transform(X_train)
        if self.f_select != None:
            train_matrix = self.train_feature_selection(train_matrix, y_train)
        # cat_matrix = self.fvectorizer.fit_transform(X_train)
        # combined_matrix = hstack([matrix, cat_matrix], format='csr')
        # train_matrix = combined_matrix
        # tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
        # X_train_tf = tf_transformer.transform(X_train_counts)
        # X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)
        return train_matrix

    def prepare_input_matrix(self, X_test):
        X_test = self.vectorizer.transform(X_test)
        if self.f_select != None:
            X_test = self.selection.transform(X_test)
        # cat_matrix = self.fvectorizer.transform(X_test)
        # combined_matrix = hstack([matrix, cat_matrix], format='csr')
        # X_new_counts = combined_matrix
        # X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        return X_test
    
    def train_feature_selection(self, X_train, y_train):
        print("feature selection...")
        self.selection = SelectKBest(score_func=chi2, k=self.f_select)
        X_train = self.selection.fit_transform(X_train, y_train)
        print("Done")
        return X_train

class MultiClassifier(object):
    
    cat_clf = dict()
    
    def __init__(self, clf):
        self.clf = clf
    
    def train(self,X,targets):
        for k,v in targets.items():
            print("\n#########################################################\n" + k + "\n")
            self.cat_clf[k] = self.clf.train(X,v)
        
    def classify(self, input):
        for cat,clf in self.cat_clf.items():
            print('\nfor ' + cat + ' classified like:\n\n')
            for k, v in self.clf.classify(input).items():
                
                print(k + " --> " + v)
        
class ClassifiersGallery(object):
    
    clf_hist = defaultdict(list)
    
    y = []
    
    vectorizer = None
    
    def __init__(self, Vect):
        self.vectorizer = Vect
        self.fvectorizer = Vect
    
    
    def gen_gallery(self, sizes):
        
        self.sizes = sizes
        t = time.strftime("%d-%m|%H:%M")
        
        fname = '/home/alejandro/Documenti/training_set.csv'  # '/home/alejandro/Documenti/Xand3cat.csv' #/home/alejandro/Documenti/VISIONATI.csv' 
        del_char = "|"
        limit = 1000000
        col_desc = 0
        col_cat = ["AREA_INTERVENTO", 'SETTORE_INTERVENTO', 'SOTTOSETTORE_INTERVENTO', 'CATEGORIA_INTERVENTO']
        
        data_utils = DataUtils()
        
        print('loading data..')
        df = data_utils.csvReader(fname, limit, del_char)
        print('Done')
        
        for campione in sizes:
            
            classifier = Classifier()
            print('\nCAMPIONE : ' + str(campione) + '\n')
            print('sampling data..')
            X, targets = data_utils.gen_XandY(df, campione, col_desc, col_cat)
            print('Done')
            
            classifier.train(X, targets)
        
        n_gram = classifier.get_n_gram()
        if n_gram != None:
            ng = "|Ngram" + str(n_gram)
        else:
            ng = ''
            
        hash_features = classifier.get_hash_features()
        if hash_features != None:
            hf = "|HashVectorizer(" + str(hash_features) + ")"
        else:
            hf = ""
            
        for cat in col_cat:
            self.clf_hist = defaultdict(list)
            for accuracy in classifier.get_accuracies():       
                self.makeClfHist(accuracy, cat)
            
            clf_names = list(sorted(self.clf_hist.keys()))
            
            plt.figure()
            histories = []
            for name in clf_names:
                histories.append(self.clf_hist[name])
            for stats in histories:
                self.plot_accuracy(sizes, stats, cat)
                ax = plt.gca()
                ax.set_ylim((0.5, 1))
            plt.legend(clf_names, loc='best')
            
            
            directory = '/home/alejandro/OpenCup_plots/' + t + ng + hf
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            plt.savefig(directory + "/" + cat + '.png')
            plt.show()
        
    def makeClfHist(self, accuracies, label):
        
        for clf, acc in accuracies[label].items():
            self.clf_hist[clf].append(acc)
            
                  
    def plot_accuracy(self, x, y, x_legend):
        
        """Plot accuracy as a function of x."""
        x = np.array(x)
        y = np.array(y)
        plt.title(x_legend)
        plt.xlabel('%s' % 'campioni')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.plot(x, y)
        # plt.show()


        
    
        
        
# clf_g = ClassifiersGallery()

n_gram = (1, 2)
hash_features = 50000
# clf_g.gen_gallery([2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000], n_gram=n_gram)

     
        
