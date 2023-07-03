# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:31:00 2023

@author: claire.dussard
"""

import mne

from functions.load_savedData import *
from handleData_subject import createSujetsData
from functions.load_savedData import *
import numpy as np
import os
import pandas as pd
from mne.time_frequency.tfr import combine_tfr

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

cond = "main"
listeTfr_cond_exclude_bad = load_tfr_data_windows(liste_rawPathMain[13:15],"alltrials",True)

#recenser les runs dispo

jetes_main = [
    [],[],[],[3],[2,4,5,6,7],[7],[],[6],[9,10],[8],[6],
    [1,6,8],[1,10],[9,10],[6,7,8,9,10],[3,6],[3,6,7],[4,10],[],[1,6],[],[9],[]
    ]

jetes_pendule = [
    [],[],[],[5],[1,7,10],[],[],[3,5,8,10],[],[5,10],[],
    [5,6],[4],[6,9],[],[9],[3,8,9],[],[],[1,6],[6],[3,9],[6,8]
    ]

jetes_mainIllusion = [
    [6],[1,3,6],[1,2],[],[5,6,8,9,10],[],[],[1,6,7,8],[6,7,8,9,10],[4,10],[1],
    [],[1,8,10],[10],[6,9],[9],[4,8,9],[4,8],[],[1,6],[],[1],[]
    ]



run_dispo_main = []
for jete in jetes_main:
    run1 = all(element in jete for element in [1,2,3,4,5])
    run2 = all(element in jete for element in [6,7,8,9,10])
    run_dispo_main.append([run1,run2])
    
run_dispo_pendule = []
for jete in jetes_pendule:
    run1 = all(element in jete for element in [1,2,3,4,5])
    run2 = all(element in jete for element in [6,7,8,9,10])
    run_dispo_pendule.append([run1,run2])

run_dispo_mainIll = []
for jete in jetes_mainIllusion:
    run1 = all(element in jete for element in [1,2,3,4,5])
    run2 = all(element in jete for element in [6,7,8,9,10])
    run_dispo_mainIll.append([run1,run2])
    
    
#modify so you get one value per run


essaisJetes_cond = jetes_main

liste_tfr_allsujets_run1 = []
liste_tfr_allsujets_run2 = []
compte_jetes = [0 for i in range(23)]
for j in range(23):#n_sujets
    print("sujet "+str(j))
    essais_jetes_suj = essaisJetes_cond[j]
    print("essais jetes")
    print(essais_jetes_suj)
    run_dispo = run_dispo_main[j]
    liste_tfr_sujet_run1 = []
    liste_tfr_sujet_run2 = []
    for i in range(10):
        if i+1 not in essais_jetes_suj:
            print("dispo")
            index = i-compte_jetes[j]
            if i+1<6:
                print("run 1 ")
                liste_tfr_sujet_run1.append(index)
            else:
                print("run 2 ")
                liste_tfr_sujet_run2.append(index) 
        else:
            compte_jetes[j] += 1 
            print("essai jete")
            index = i-compte_jetes[j]
 
        print(liste_tfr_sujet_run1)
        print(liste_tfr_sujet_run2)
    liste_tfr_allsujets_run1.append(liste_tfr_sujet_run1)
    liste_tfr_allsujets_run2.append(liste_tfr_sujet_run2)
    
    
#compte_jetes = [0 for i in range(23)]

def gd_average_allEssais_V3(liste_tfr_cond,get_all_suj,baseline,index_essais_run1,index_essais_run2):
    dureePreBaseline = 3 #3
    dureePreBaseline = - dureePreBaseline
    dureeBaseline = 2.0 #2.0
    valeurPostBaseline = dureePreBaseline + dureeBaseline
    baseline = (dureePreBaseline, valeurPostBaseline)
    all_listes_run1 = []
    all_listes_run2 = []
    for i in range(len(liste_tfr_cond)):
        print("num sujet" + str(i))
        liste_tfr_suj = liste_tfr_cond[i]
        indexes_run1 = index_essais_run1[i]
        print('run 1')
        essais_run1 = [liste_tfr_suj[i] for i in indexes_run1]
        essais_run1 = [item.average() for item in essais_run1]
        av_essais_run1 = combine_tfr(essais_run1,'equal')
        bl_essais_run1 = av_essais_run1.apply_baseline(baseline=baseline, mode='logratio')
        all_listes_run1.append(bl_essais_run1)
        print('run 2')
        indexes_run2 = index_essais_run2[i]
        essais_run2 = [liste_tfr_suj[i] for i in indexes_run2]
        essais_run2 = [item.average() for item in essais_run2]
        av_essais_run2 = combine_tfr(essais_run2,'equal')
        bl_essais_run2 = av_essais_run2.apply_baseline(baseline=baseline, mode='logratio')
        all_listes_run2.append(bl_essais_run2)
        return bl_essais_run1,bl_essais_run2
        
bl_essais_run1,bl_essais_run2 = gd_average_allEssais_V3(listeTfr_cond_exclude_bad,True,True,liste_tfr_allsujets_run1[13:15],liste_tfr_allsujets_run2[13:15])



av_essais_run1 = [item.average() for item in essais_run1]
    