
'''
Created on Dec 2, 2016

@author: daniele
'''



import pprint
import re

import pymongo
from pymongo.mongo_client import MongoClient


class ConnectDBMongo(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        client = MongoClient('192.168.7.41', 27017)
        db = client.test
        self.collection = db.twitter
        #post_id = posts.insert_one(post).inserted_id 
        #print(post_id)
        #print(posts.find_one({"author": 2}))
        
    def inserTwitter(self,twitter):
        posts = self.collection
        post_id = posts.insert_one(twitter).inserted_id 
    
    def searchWord(self,word,field='text'):
        posts = self.collection
        pat = re.compile(r''+word, re.I)
        for post in posts.find({field:  {'$regex': pat} }):
            pprint.pprint(post)
         
if __name__ == '__main__':
    a = ConnectDBMongo()
    a.searchWord('no')
