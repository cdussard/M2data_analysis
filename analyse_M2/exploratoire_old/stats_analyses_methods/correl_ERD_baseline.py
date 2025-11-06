#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:22:27 2022

@author: claire.dussard
"""
#compute table to correlate baseline and ERD value

import pandas as pd
import numpy as np

table_ERD = pd.read_csv("data/correl/dataParEssaiFB_sujets_8_30.csv")

#num_sujets = [-1,0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] #-1 = moyenne
len(num_sujets)

table_bl_main = pd.read_csv("data/correl/mainBaseline8-30_23sujets.csv")
table_bl_main.numSujet = [int(num) for num in num_sujets]
table_bl_main.rename(columns={"essai": "initiales"})
table_bl_main = table_bl_main.replace('E', 'e', regex=True).replace(',', '.', regex=True)
table_bl_main = table_bl_main.apply(pd.to_numeric, args=('coerce',))
table_bl_main.essai = initiales
table_bl_main.numSujet = [int(num) for num in num_sujets]
table_bl_main.rename(columns={"essai": "initiales"})

table_bl_mainIllusion = pd.read_csv("data/correl/mainIllusionBaseline8-30_23sujets.csv")#,dtype=np.float64)
initiales = table_bl_mainIllusion.essai
table_bl_mainIllusion = table_bl_mainIllusion.replace('E', 'e', regex=True).replace(',', '.', regex=True)
# convert notation to the one pandas allows
table_bl_mainIllusion = table_bl_mainIllusion.apply(pd.to_numeric, args=('coerce',))
table_bl_mainIllusion.essai = initiales
table_bl_mainIllusion.numSujet = [int(num) for num in num_sujets] 
table_bl_mainIllusion.rename(columns={"essai": "initiales"})

table_bl_pendule = pd.read_csv("data/correl/penduleBaseline8-30_23sujets.csv")
table_bl_pendule = table_bl_pendule.replace('E', 'e', regex=True).replace(',', '.', regex=True)
table_bl_pendule = table_bl_pendule.apply(pd.to_numeric, args=('coerce',))
table_bl_pendule.essai = initiales
table_bl_pendule.numSujet = [int(num) for num in num_sujets]
table_bl_pendule.rename(columns={"essai": "initiales"})


list_values_BL_pendule = []#5 valeurs par sujet
list_values_BL_main = []
list_values_BL_mainIllusion = []

list_values_ERD_pendule = []#5 valeurs par sujet
list_values_ERD_main = []
list_values_ERD_mainIllusion = []

#plus efficace que peupler un dataframe apparemment : https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it

for i in range(1,24):#on passe la moyenne
    print("sujet "+str(table_bl_main.numSujet[i]))
    #3 cond
    #cond 1
    BL_main = table_bl_main.loc[i][2:].tolist()
    BL_mainIllusion = table_bl_mainIllusion.loc[i][2:].tolist()
    BL_pendule = table_bl_pendule.loc[i][2:].tolist()
    ERD_main = table_ERD[(table_ERD.num_sujet==i-1) & (table_ERD.FB_int == 1)]["MedianeERD"].tolist()
    ERD_mainIllusion = table_ERD[(table_ERD.num_sujet==i-1) & (table_ERD.FB_int == 4)]["MedianeERD"].tolist()
    ERD_pendule = table_ERD[(table_ERD.num_sujet==i-1) & (table_ERD.FB_int == 2)]["MedianeERD"].tolist()
    print(ERD_pendule)
    print(BL_pendule)
    list_values_BL_pendule.append(BL_pendule)
    list_values_ERD_pendule.append(ERD_pendule)
    list_values_BL_main.append(BL_main)
    list_values_ERD_main.append(ERD_main)
    list_values_ERD_mainIllusion.append(ERD_mainIllusion)
    list_values_BL_mainIllusion.append(BL_mainIllusion)
    
from math import log10
#can't flatten the sublists (strings) 
#les ERD OK, les BL non
#flatten ERD
list_values_ERD_main = [item for sublist in list_values_ERD_main for item in sublist]
list_values_ERD_main = [log10(item) for item in list_values_ERD_main]
list_values_ERD_mainIllusion = [item for sublist in list_values_ERD_mainIllusion for item in sublist]
list_values_ERD_mainIllusion = [log10(item) for item in list_values_ERD_mainIllusion]
list_values_ERD_pendule = [item for sublist in list_values_ERD_pendule for item in sublist]
list_values_ERD_pendule = [log10(item) for item in list_values_ERD_pendule]

allERD = list_values_ERD_pendule + list_values_ERD_main + list_values_ERD_mainIllusion
#flatten BL
list_values_BL_main = [item for sublist in list_values_BL_main for item in sublist]
list_values_BL_mainIllusion = [item for sublist in list_values_BL_mainIllusion for item in sublist]
list_values_BL_pendule = [item for sublist in list_values_BL_pendule for item in sublist]

allBL = list_values_BL_pendule + list_values_BL_main + list_values_BL_mainIllusion

#baselines 
#deal with Nan characters before conversion : convert to -1
for i in range(len(list_values_BL_main)):
    sublist = list_values_BL_main[i]
    sublist = ['-1' if x=='_' else x for x in sublist]
    list_values_BL_main[i] = [float(el) for el in sublist]#PB : le grand E n'est pas reconnu comme ecriture scientifique donc pas convertible en float
for i in range(len(list_values_BL_mainIllusion)):
    sublist = list_values_BL_mainIllusion[i]
    sublist = ['-1' if x=='_' else x for x in sublist]
    list_values_BL_mainIllusion[i] = sublist
for i in range(len(list_values_BL_pendule)):
    sublist = list_values_BL_pendule[i]
    sublist = ['-1' if x=='_' else x for x in sublist]
    list_values_BL_pendule[i] = sublist
    


df_correl = pd.DataFrame({'condition': np.tile([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],23),
                   'sujet': np.repeat([0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],15),
                      'ERD':allERD,
                      'Baseline':allBL})

df_correl.to_csv("data/correl/output/df_correl.csv")
    
    
    
