'''
Created on 05 dic 2016

@author: alejandro
'''
import pandas
from pandas.core.frame import DataFrame

from utils.DataUtils import DataUtils


fname = '/home/alejandro/Scrivania/TOTALE.csv'  # '/home/alejandro/Documenti/Xand3cat.csv' #/home/alejandro/Documenti/VISIONATI.csv' 
del_char = "|"
limit = 1000000
#open(file, mode, buffering, encoding, errors, newline, closefd, opener)
c=''

with open('/home/alejandro/Scrivania/TOTALE.csv','r',encoding = 'latin') as f1:
    for row in f1:
        c_quality = row.split('|')[68]
        if c_quality == "Il corredo informativo di questo CUP è stato visionato, almeno parzialmente":
            c+=row+'\n'
        
with open('/home/alejandro/Scrivania/VALIDATI.txt','w') as f2:
    f2.write(c)
    
df = pandas.read_csv('/home/alejandro/Scrivania/VALIDATI.csv')