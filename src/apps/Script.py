'''
Created on 19 dic 2016

@author: alejandro
'''
c=0
with open('/home/alejandro/Scrivania/TOTALE.csv','r',encoding = 'latin') as f1:
    with open('/home/alejandro/Scrivania/check.txt','w') as f2:
        for row in f1:
            c+=1;
            print(c)
            if(len(row.split('|')) !=72):
                print(len(row.split('|')))
                f2.write(row)
print(c,"FINE")