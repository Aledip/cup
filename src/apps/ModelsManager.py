# -*- coding: utf-8 -*-
'''
Created on 06 ott 2016

@author: Alejandro
'''
from _collections import defaultdict, OrderedDict
import os
import time

from scipy.sparse.construct import hstack
from sklearn.cross_validation import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.perceptron import Perceptron
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import numpy as np
from utils.DataUtils import DataUtils


class Accuracy(object):
    
    label = ''
    clf = ''
    accuracies = []
        
    def __init__(self, label, clf, accuracies):
        self.label = label
        self.clf = clf
        self.accuracies = accuracies
    
class Classifier(object):
        
    def get_labels(self):
        return self.labels
    
    def get_accuracies(self):
        return self.accuracies
    
    def get_n_gram(self):
        return self.n_gram
    
    def get_hash_features(self):
        return self.hash_features


    path = "/home/alejandro/models/"
    labels = dict()
    count_vect = None
    count_vect_cat = None
    tfidf_transformer = None
    accuracies = []
    n_gram = None
    hash_features = None
    
    
    def train(self, X, targets, n_gram=None, hash_features=None):
        
        self.n_gram = n_gram
        self.hash_features = hash_features
        dict1 = OrderedDict()
        for label in targets:
            models = []
            
            print('\ngenerating training_set for ' + label + '..')
            X_train, X_test, y_train, y_test = train_test_split(X, targets[label], train_size=0.70, random_state=42)
            print('Done\n')
            
            params = {'alpha': [0.01, 0.001 , 0.0001], }
                        
            
            if(hash_features != None):
                self.count_vect = HashingVectorizer(non_negative=True, n_features=self.hash_features, ngram_range=self.n_gram)
                self.count_vect_cat = HashingVectorizer(non_negative=True, n_features=self.hash_features, ngram_range=self.n_gram)
            else:
                self.count_vect = CountVectorizer(ngram_range=self.n_gram)
                self.count_vect_cat = CountVectorizer(ngram_range=self.n_gram)
             
            self.tfidf_transformer = TfidfTransformer(use_idf=False)
            
            models.append(MultinomialNB(alpha=0.001))
            models.append(PassiveAggressiveClassifier(n_iter=10, n_jobs=-1))
            models.append(SGDClassifier(loss='perceptron', alpha=0.001, n_iter=100, n_jobs=-1))
            models.append(Perceptron(alpha=0.001, n_iter=100, n_jobs=-1, random_state=1))
            models.append(RandomForestClassifier(n_estimators=20, n_jobs=-1))
    
            # self.models['AB'] = AdaBoostClassifier(base_estimator=self.models['NB'], n_estimators=100)
            # models.append( VotingClassifier(estimators=[('NB', models[0]), ('RF', models[4]), ('LR', models[5])], voting='soft', weights=[2,2,1]) )
            # models.append(GridSearchCV(estimator=models['NB'], param_grid=params, cv=5))
            
            t_start_vect = time.time()   
            X_train_tfidf = self._prepare_matrix(X_train)
            print(str(round(time.time() - t_start_vect, 3)) + "s for Vectorization with " + type(self.count_vect).__name__ + "\n")
            
            for m in models:
                t_start = time.time()
                m.fit(X_train_tfidf, y_train)
                print(str(round(time.time() - t_start, 3)) + "s for training " + type(m).__name__)
                
            self.labels[label] = models
            accuracy = self.get_accuracy(X_test, y_test, label)
            print(accuracy)
            
            dict1[label] = accuracy
        self.accuracies.append(dict1)
        
    
    def set_models(self, models):
        self.models = models
    
    def get_models(self):
        return self.models
    
    def add_element(self, x_desc, y_label):
        
        for m in self.models:
            m.partial_fit(x_desc, y_label)
        
        
    def save(self):
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        name = "modelli_addestrati.pkl"
        print ("saving models...")
        t_start_vect = time.time() 
        joblib.dump([self.labels, self.count_vect, self.count_vect_cat, self.tfidf_transformer], self.path + name, compress=3)
        print(str(round(time.time() - t_start_vect, 3)) + "s for save models")
        print("Done")
        
      
    def load(self):
        
        name = "modelli_addestrati.pkl"
        print("loading models...")
        t_start_vect = time.time()
        self.labels, self.count_vect, self.count_vect_cat, self.tfidf_transformer = joblib.load(self.path + name)
        print(str(round(time.time() - t_start_vect, 3)) + "s for load models")
        print("Done")
     
    def get_accuracy(self, X_test, y_test, label):
        accuracies = dict()
        predictions = self.predict_new(X_test, label)
        for m in predictions:
            accuracies[m] = round(np.mean(predictions[m] == y_test), 2)
        return accuracies
    
    def classify(self, X_test, label):
        clf_res = dict()
        predictions = self.predict_new(X_test, label)
        for m in predictions:
            clf_res[m] = predictions[m][0]
        return clf_res
    
    def predict_new(self, X_test, label):
        test_matrix = self.prepare_test_matrix(X_test)
        predictions = dict()
        for model in self.labels[label]:
            predictions[type(model).__name__] = model.predict(test_matrix)
        return predictions
            
    
    def _prepare_matrix(self, X_train):
        
        matrix = self.count_vect.fit_transform(X_train)
        cat_matrix = self.count_vect_cat.fit_transform(X_train)
        combined_matrix = hstack([matrix, cat_matrix], format='csr')
        X_train_counts = combined_matrix
        # tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
        # X_train_tf = tf_transformer.transform(X_train_counts)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)
        return X_train_tfidf

    def prepare_test_matrix(self, X_test):
        matrix = self.count_vect.transform(X_test)
        cat_matrix = self.count_vect_cat.transform(X_test)
        combined_matrix = hstack([matrix, cat_matrix], format='csr')
        X_new_counts = combined_matrix
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        return X_new_tfidf
    
    
    
        
class ClassifiersGallery(object):
    
    clf_hist = defaultdict(list)
    
    y = []
    
    
    def gen_gallery(self, sizes, n_gram=None, hash_features=None):
        
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
            
            classifier.train(X, targets, n_gram=n_gram, hash_features=hash_features)
        
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
        
    
        
        
#clf_g = ClassifiersGallery()

n_gram = (1, 2)
hash_features = 50000
#clf_g.gen_gallery([2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000], n_gram=n_gram)

     
        
