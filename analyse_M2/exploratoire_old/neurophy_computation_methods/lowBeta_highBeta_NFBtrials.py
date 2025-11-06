# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:15:16 2023

@author: claire.dussard
"""

#load les TFR sans Baseline

import os 
import seaborn as sns
import pathlib
import mne
#necessite d'avoir execute handleData_subject.py, et load_savedData avant 
import numpy as np 
# importer les fonctions definies par moi 
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)


liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)


liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule[0:2],"allTrials",True)
liste_tfrMain = load_tfr_data_windows(liste_rawPathMain[0:2],"allTrials",True)
liste_tfrMainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion[0:2],"allTrials",True)

#filtrer en 8-30 puis 13-20 et 21-30 (en copiant les data sinon tu modifies in place)

for tfr in liste_tfrPendule+liste_tfrMain+liste_tfrMainIllusion:
    tfr.pick_channels(["C3"])
    
import pandas as pd
#calculer puissance en C3 en moyennant sur le temps pendant les 2s de BL ou pendant les 22s de NFB


def append_mean_value(liste_tfr,tmin,tmax,fmin,fmax,df,nomFB,numSujet):
    tfr_sujet_cond = liste_tfr.copy().crop(fmin=fmin,fmax=fmax,tmin=tmin,tmax=tmax)
    if tmax<0:
        time = "BL"
    else:
        time = "NF"
    numTrial = 0
    for tfr in tfr_sujet_cond:
        dataBaseline = tfr
        value = np.mean(dataBaseline)
        dictionnaire = {
             "numSujet":[numSujet],
              "numTrial":[numTrial],
              "FB":[nomFB],
              "bande":[str(fmin)+"-"+str(fmax)],
              "time":[time],
              "power":[value]
             }
        dataTrial =  pd.DataFrame.from_dict(dictionnaire)
        df = df.append(dataTrial,ignore_index=True)
        #pd.concat(d,ignore_index=True)
        numTrial += 1
    return df

full_dataframe = pd.DataFrame(columns=["numSujet","numTrial","FB","bande","time","power"])

for numSujet in range(23):
    liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule[numSujet:numSujet+1],"allTrials",True)
    liste_tfrMain = load_tfr_data_windows(liste_rawPathMain[numSujet:numSujet+1],"allTrials",True)
    liste_tfrMainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion[numSujet:numSujet+1],"allTrials",True)
    liste_tfr = liste_tfrPendule
    for FB,liste_tfr in zip(["pendule","main","mainIllusion"],[liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion]):
        for tfr_sujet_cond in liste_tfr:
            full_dataframe = append_mean_value(tfr_sujet_cond,-3,-1,8,30,full_dataframe,FB,numSujet)
            full_dataframe = append_mean_value(tfr_sujet_cond,-3,-1,13,20,full_dataframe,FB,numSujet)
            full_dataframe = append_mean_value(tfr_sujet_cond,-3,-1,21,30,full_dataframe,FB,numSujet)
            full_dataframe = append_mean_value(tfr_sujet_cond,2.5,25.5,8,30,full_dataframe,FB,numSujet)
            full_dataframe = append_mean_value(tfr_sujet_cond,2.5,25.5,13,20,full_dataframe,FB,numSujet)
            full_dataframe = append_mean_value(tfr_sujet_cond,2.5,25.5,21,30,full_dataframe,FB,numSujet)


#tout stocker dans un dataframe avec en colonnes num_sujet, numTrial, power,"type":NF/BL,bande;conditionFB

full_dataframe.to_csv("../csv_files/data_power_NF_BL.csv")
