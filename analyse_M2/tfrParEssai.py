#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 18:25:07 2022

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

 
#load previous data
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
  
#==========acceder aux essais jetes par sujet============
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
        
def get_30essais_sujet_i(num_sujet,freqMin,freqMax,pasFreq,sujets_epochs_jetes_main,sujets_epochs_jetes_mainIllusion,sujets_epochs_jetes_pendule):
    freqs = np.arange(3, 85, pasFreq)
    n_cycles = freqs 
    i = num_sujet
    
    #pendule
    epochs_sujet = EpochDataPendule[i]
    print(epochs_sujet)
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...") #decim= 5 verifier si resultat pareil qu'avec down sampling
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False,average=False)#AVERAGE = FALSE : 1 par essai
    print("computing power...")
    power_sujet_pendule = power_sujet
    
    #main
    epochs_sujet = EpochDataMain[i]
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...") #decim= 5 verifier si resultat pareil qu'avec down sampling
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False,average=False)#AVERAGE = FALSE : 1 par essai
    print("computing power...")
    power_sujet_main = power_sujet
    
    #mainIllusion
    epochs_sujet = EpochDataMainIllusion[i]
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...") #decim= 5 verifier si resultat pareil qu'avec down sampling
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False,average=False)#AVERAGE = FALSE : 1 par essai
    print("computing power...")
    power_sujet_mainIllusion = power_sujet
    
    dixEssais_TFR_test_pendule = power_sujet_pendule#on ne calcule qu'une seule puissance
    dixEssais_TFR_test_main = power_sujet_main
    dixEssais_TFR_test_mainIllusion = power_sujet_mainIllusion
    
    dureePreBaseline = 3 #3
    dureePreBaseline = - dureePreBaseline
    dureeBaseline = 2.0 #2.0
    valeurPostBaseline = dureePreBaseline + dureeBaseline
    baseline = (dureePreBaseline, valeurPostBaseline)
    
    fig,axs = plt.subplots(3,10)
    vmin = -0.75
    vmax = 0.75
    essaisJetes_main_sujet = sujets_epochs_jetes_main[i]
    essaisJetes_pendule_sujet = sujets_epochs_jetes_pendule[i]
    essaisJetes_mainIllusion_sujet = sujets_epochs_jetes_mainIllusion[i]
    mode = 'zscore'
    #Type = EpochsTFR : pas de fonction plot ?? non, donc on convertit artificiellement en average  :https://mne.discourse.group/t/how-to-plot-epochstfr-images/3070/3
    delta = 0  
    for i in range(10):
        if i+1 not in essaisJetes_pendule_sujet:#gerer les epochs jetes dans l'affichage
        #dixEssais_TFR_test[i].average().plot(picks="C3",baseline=(-3,-1),mode='logratio',fmax=40,vmin=-0.3,vmax=0.3)
        #dixEssais_TFR_test[i].average().plot_topomap(baseline=(-3,-1),mode='logratio',fmin=8,fmax=30,tmin=2,tmax=25)#,vmin=-0.3,vmax=0.3)
            dixEssais_TFR_test_pendule[i-delta].average().plot_topomap(baseline=baseline,mode=mode,fmin=freqMin,fmax=freqMax,tmin=2,tmax=25,axes=axs[0,i],vmin=vmin,vmax=vmax)
        else:
            print("essai jete num"+str(i+1))
            delta += 1
            
    delta = 0   
    for i in range(10):
        if i+1 not in essaisJetes_main_sujet:
            dixEssais_TFR_test_main[i-delta].average().plot_topomap(baseline=baseline,mode=mode,fmin=freqMin,fmax=freqMax,tmin=2,tmax=25,axes=axs[1,i],vmin=vmin,vmax=vmax)
        else:
            print("essai jete num"+str(i+1))
            delta += 1
    delta = 0   
    for i in range(10):  #colorbar = False pour ne pas plot toutes les colorbar
        if i+1 not in essaisJetes_mainIllusion_sujet:
            dixEssais_TFR_test_mainIllusion[i-delta].average().plot_topomap(baseline=baseline,mode=mode,fmin=freqMin,fmax=freqMax,tmin=2,tmax=25,axes=axs[2,i],vmin=vmin,vmax=vmax)
        else:
            print("essai jete num"+str(i+1))
            delta += 1
    return True

get_30essais_sujet_i(num_sujet=22,freqMin=3,freqMax=85,pasFreq=1,
                     sujets_epochs_jetes_main=jetes_main,sujets_epochs_jetes_mainIllusion=jetes_mainIllusion,sujets_epochs_jetes_pendule=jetes_pendule)
raw_signal.plot(block=True)

for i in range(6,12):#5 par 5 pour ne pas saturer la RAM
    get_30essais_sujet_i(num_sujet=i,freqMin=8,freqMax=30,pasFreq=1,
                     sujets_epochs_jetes_main=jetes_main,sujets_epochs_jetes_mainIllusion=jetes_mainIllusion,sujets_epochs_jetes_pendule=jetes_pendule)
raw_signal.plot(block=True)
    

