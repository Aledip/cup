'''
Created on 07 ott 2016

@author: Alejandro
'''

import collections

import pandas as pd
from utils.TextUtils_oop import TextUtils


class DataUtils(object):
    
    def csvReader(self,fname,limit,del_char):
        '''leggo il file csv con limite e delimitatore,e lo carico come dataframe'''
        df = pd.read_csv(fname,nrows=limit,delimiter=del_char,error_bad_lines=False,encoding = "ISO-8859-1")
        return df
    
    def gen_XandY(self,df,nsample,X_column,y_columns):
        
        
        sample = df.sample(n=nsample)
        
        matrix = sample.as_matrix()
        
        X = matrix[:,X_column]
        
        #  natura intervento 34 - tipologia_intervento 36 - area_intervento 37 - settore_intervento 39 - sottosettore_intervento 41 - categoria_intervento 43
        y_dict = collections.OrderedDict()
        for y_column in y_columns:
            y_dict[y_column] = sample.loc[:,y_column] #[as_matrix()[:,y_column] # .tolist()

        X = TextUtils().norm_str_array(X)
        
        return X,y_dict    
    
    def genCategory(self,fname,del_char,col_cat):
        insieme = set()
        with open(fname,'r') as f:
            for line in f:
                insieme.add(line.split(del_char)[col_cat])
        alpha_category = list(insieme).sort()
        return alpha_category
    
    def fileReaderRanged(self,fname,start,end):
        with open(fname,'r') as f:
            c=0
            lista=[]
            for line in f:
                if start<c<=end:
                    lista.append(line)
                c+=1
        return lista
    
    
    