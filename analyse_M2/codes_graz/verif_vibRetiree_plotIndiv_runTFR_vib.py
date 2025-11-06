# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:16:53 2024

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
import scipy
from scipy import io
from scipy.io import loadmat
from functions_graz import *

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

#listeTfr_cond_exclude_bad = load_tfr_data_windows(liste_rawPathMainIllusion[0:1],"alltrials",True)[0]




#============== essayer d'exclure les parties avec vibration ========

#recenser les runs dispo

jetes_mainIllusion = [
    [6],[1,3,6],[1,2],[],[5,6,8,9,10],[],[],[1,6,7,8],[6,7,8,9,10],[4,10],[1],
    [],[1,8,10],[10],[6,9],[9],[4,8,9],[4,8],[],[1,6],[],[1],[]
    ]

def get_TFR_cond(jetes_cond,save,cond,listeTfr_cond_exclude_bad):
    #get dispo
    liste_tfr_allsujets_run1,liste_tfr_allsujets_run2 = getDispo_cond(jetes_cond)
    #extract trials per run
    all_listes_run1,all_listes_run2 = gd_average_allEssais_V3(listeTfr_cond_exclude_bad,True,True,liste_tfr_allsujets_run1,liste_tfr_allsujets_run2)
    return all_listes_run1,all_listes_run2

listeTfr_cond_exclude_bad = load_tfr_data_windows(liste_rawPathMainIllusion[0:4],"alltrials",True)

all_listes_run1,all_listes_run2 = get_TFR_cond(jetes_mainIllusion,True,"mainIllusion",listeTfr_cond_exclude_bad)
    

for tfr in listeTfr_cond_exclude_bad:
    tfr.average().plot(picks="C3",fmax=40)
    
listeTfr_cond_exclude_bad[3].crop(tmin=2.5,tmax=26.5)
data_theta = listeTfr_cond_exclude_bad[3].average()._data[:, :5, :]

plt.plot(listeTfr_cond_exclude_bad[3].times,np.mean(data_theta[12],axis=0))
    
t_Startvib_nf = [6.2,12.4,18.6,24.8]
durVib = 2 #•1.3 sur la derniere 
fEch = 250
intDurVib = durVib*fEch
intVibss = [int((val*fEch)-2.5*fEch) for val in t_Startvib_nf]
endVibs = [startvib + intDurVib for startvib in intVibss]
segments_to_exclude = []
for i in range(len(intVibss)):
    couple = (intVibss[i],endVibs[i])
    segments_to_exclude.append(couple)
    
    
mask = np.zeros(data_theta.shape[2], dtype=bool)
for segment in segments_to_exclude:
    start, end = segment
    mask[start:end+1] = True

# Invert the mask to select the values outside of the segments
inverted_mask = ~mask

# Use the mask to select the values outside of the segments
values_outside_segments =data_theta[:, :, inverted_mask]


vals = np.mean(values_outside_segments,axis=2) 

plt.plot(listeTfr_cond_exclude_bad[3].times,np.mean(data_theta[12],axis=0))
plt.plot(np.arange(1,4073),np.mean(values_outside_segments[12],axis=0))
 
    