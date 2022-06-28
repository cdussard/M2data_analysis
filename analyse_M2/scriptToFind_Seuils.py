# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:53:44 2022

@author: claire.dussard
"""

import pandas as pd 
import numpy as np

import pandas as pd 
import numpy as np
sujets  = ["S0"+str(i) for i in range(10)]
sujets_suite  = ["S"+str(i) for i in range(10,25)]
suj = sujets + sujets_suite
for i in range(24):
    print(suj[i])
    path = "./sujets_copyComputeMedian/"+suj[i]+"/csv_files/"
    sep = ","
    tableau = pd.read_csv(path+'mediane.csv',sep)
    medianeC4 = tableau.iloc[:,2]
    medianeGlobale = round(np.median(medianeC4),1)
    ecartType = round(np.std(medianeC4),1)
    d = {'mediane': [medianeGlobale], 'ecart-type': [ecartType]}
    df = pd.DataFrame(data=d)
    print(df)
    
    #dfResult=df.to_csv(path+"res_unity.csv",index=False,sep = ',')


tableau = None
sujets  = ["S0"+str(i) for i in range(10)]
sujets_suite  = ["S"+str(i) for i in range(10,25)]
suj = sujets + sujets_suite

mediane_values = []
sd_values = []
     
for i in range(25):
    print(suj[i])
    path = "./sujets_copyComputeMedian/"+suj[i]+"/csv_files/"
    sep = ","
    try :
        tableau = pd.read_csv(path+'saveMediane.csv',sep)
        print("read saveMediane.csv")
    except Exception:
        print('Another Error!!', path+'saveMediane.csv')#sometimes saveMediane.csv exists as a file but is null
        tableau = pd.read_csv(path+'mediane.csv',sep)
        print("read mediane.csv")
        medianeC4 = tableau.iloc[:,2]
    except FileNotFoundError:
        print('File not found!!', path+'saveMediane.csv')
        #if len(tableau)<15:
        print("took all values")
        tableau = pd.read_csv(path+'mediane.csv',sep)
        print("read mediane.csv")
        medianeC4 = tableau.iloc[:,2]
    else:
        medianeC4 = tableau.iloc[:,0]
    print(medianeC4)
    medianeGlobale = round(np.median(medianeC4),1)
    ecartType = round(np.std(medianeC4),1)
    d = {'mediane': [medianeGlobale], 'ecart-type': [ecartType]}
    df = pd.DataFrame(data=d)
    print(df)
    mediane_values.append(medianeGlobale)
    sd_values.append(ecartType)
    tableau = None


pd.DataFrame(mediane_values).to_csv("./mediane_sujets.csv")
pd.DataFrame(sd_values).to_csv("./sd_sujets.csv")
