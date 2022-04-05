#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:47:16 2022

@author: claire.dussard
"""
import os 
import seaborn as sns
import pathlib
import mne
#necessite d'avoir execute handleData_subject.py, et load_savedData avant 
import numpy as np 
# importer les fonctions definies par moi 
from handleData_subject import createSujetsData
from functions.load_savedData import *

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../../../")
lustre_data_dir = "iss02/cenir/analyse/meeg/BETAPARK/_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)


liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

EpochDataMain = load_data_postICA_postdropBad(liste_rawPathMain,"") 

EpochDataPendule = load_data_postICA_postdropBad(liste_rawPathPendule,"")

EpochDataMainIllusion = load_data_postICA_postdropBad(liste_rawPathMainIllusion,"")

#===================set montage===IMPORTANT!!!!=======================
montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in EpochDataMain:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
for epochs in EpochDataPendule:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
for epochs in EpochDataMainIllusion:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
        
liste_power_sujets = []
freqs = np.arange(3, 85, 1)
n_cycles = freqs 
i = 0
EpochData = EpochDataPendule

for epochs_sujet in EpochData:
    print("========================\nsujet"+str(i))
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...")
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,average=False)
    print("computing power...")
    liste_power_sujets.append(power_sujet)
    i += 1

liste_power_sujets_main = liste_power_sujets
liste_power_sujets_pendule = liste_power_sujets
liste_power_sujets_mainIllusion = liste_power_sujets

save_tfr_data(liste_power_main,rawPath_main_sujets,"")

save_tfr_data(liste_power_pendule,rawPath_pendule_sujets,"")

save_tfr_data(liste_power_mainIllusion,rawPath_mainIllusion_sujets,"")