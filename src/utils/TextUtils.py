# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 17:36:49 2016

@author: Alejandro
"""

from nltk.corpus import stopwords

import numpy as np

class TextUtils(object):
    
    def noalpha(self,s):
        noa = ''
        for c in s:
            if not (c in noa or c.isalpha()):
                noa += c
        return noa
       
    
    def descCleaner(self,desc):
        """elimina i caratteri speciali dalla descrizione"""
        for c in self.noalpha(desc):
            desc=desc.replace(c,' ')
        return desc
    
    def arrayCleaner(self,desc_array):
        '''pulisce un array di descrizioni dai caratteri speciali'''
        desc_list=[]
        for d in desc_array:
            desc = self.descCleaner(d)
            desc_list.append(desc.lower())
        return np.array(desc_list)
        
    def stopWordsCleaner(self,descriptions):
        '''restituisce un array delle descrizioni senza le stopwords e senza 
        le parole minori o uguali di 3 lettere,input array di descrizioni'''
    
        lista=[]
        stop_words= stopwords.words('italian')
        for description in descriptions:
            words=[]
            for word in description.split():
                #voglio le parole di lunghezza maggiore di 3 o xyz che non sono stopword
                if word not in stop_words and len(word)>2:
                    words.append(word)
                    
            lista.append(" ".join(words))
        return np.array(lista)
        
    def norm_str_array(self,str_array):
        clean_str_array = self.stopWordsCleaner(self.arrayCleaner(str_array))
        return clean_str_array
        
    def norm_str(self,s):
        lista = []
        s = self.descCleaner(s)
        lista.append(s)
        return self.stopWordsCleaner(lista)
