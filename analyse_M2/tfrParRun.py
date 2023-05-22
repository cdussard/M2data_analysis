# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 18:11:48 2022

@author: claire.dussard
"""

from functions.load_savedData import *
from handleData_subject import createSujetsData
from functions.load_savedData import *
#from functions.frequencyPower_displays import *
import numpy as np
import os
import pandas as pd

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
def load_indivEpochData(index_sujet,liste_rawPath):
    epoch_sujet = load_data_postICA_postdropBad_windows(liste_rawPath[index_sujet:index_sujet+1],"",True)[0]
    montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
    if epoch_sujet!=None:
        epoch_sujet.set_montage(montageEasyCap)
    return epoch_sujet
def compute_condition_power_runs(i,liste_raw,freqs,n_cycles):
    print("je suis compute_condition_power")
    epochs_sujet = load_indivEpochData(i,liste_raw)
    epochs_sujet.pick_channels(["C3"])
    print(epochs_sujet)
    print("downsampling...") #decim= 5 verifier si resultat pareil qu'avec down sampling
    epochs_sujet = epochs_sujet.resample(250., npad='auto') 
    print("computing power...")
    epochs_5Premiers_runs = epochs_sujet[0:5]
    epochs_5derniersRuns = epochs_sujet[6:]
    power_sujet_debut = mne.time_frequency.tfr_morlet(epochs_5Premiers_runs,freqs=freqs,n_cycles=n_cycles,return_itc=False,average=True)#AVERAGE = FALSE : 1 par essai
    power_sujet_fin = mne.time_frequency.tfr_morlet(epochs_5derniersRuns,freqs=freqs,n_cycles=n_cycles,return_itc=False,average=True)#AVERAGE = FALSE : 1 par essai
    return power_sujet_debut,power_sujet_fin

def compute_val(power,fmin,fmax,tmin,tmax,doBaseline):
    if doBaseline:
        power.apply_baseline(mode='logratio',baseline=(-3,-1))
    power.crop(fmin=fmin,fmax=fmax, tmin=tmin,tmax=tmax)
    power_freq = np.mean(power.data,axis=1)
    power_val = np.mean(power_freq.data,axis=1)
    power_val = np.mean(power_val ,axis=0)
    return power_val

i = 0
my_cmap = discrete_cmap(13, 'RdBu_r')
fig,axs = plt.subplots(3,10)
vmin = -0.75
vmax = 0.75
tmin = 2
tmax = 26.5
baseline = (-3,-1)
mode = 'logratio'    
freqs = np.arange(3, 40, 1)
n_cycles = freqs 


matrice_data_12_15 = np.zeros((23,6))
matrice_data_8_30 = np.zeros((23,6))

for i in range(23):
    listes = [liste_rawPathPendule,liste_rawPathMain,liste_rawPathMainIllusion]
    for j in range(len(listes)):
        liste = listes[j]
        power_sujet_debut,power_sujet_fin = compute_condition_power_runs(i,liste,freqs,n_cycles)
        #run 1
        val_8_30_run1 = compute_val(power_sujet_debut,8,30,tmin,tmax,True)
        val_12_15_run1 = compute_val(power_sujet_debut,12,15,tmin,tmax,False)
        print(val_12_15_run1)
        print(val_12_15_run2)
        print(val_8_30_run1)
        print(val_8_30_run2)
        #run 2
        val_8_30_run2 = compute_val(power_sujet_fin,8,30,tmin,tmax,True)
        val_12_15_run2 = compute_val(power_sujet_fin,12,15,tmin,tmax,False)
        print(val_12_15_run1)
        print(val_12_15_run2)
        print(val_8_30_run1)
        print(val_8_30_run2)
        matrice_data_12_15[i,2*j] = val_12_15_run1
        matrice_data_12_15[i,(1+2*j)] = val_12_15_run2
        matrice_data_8_30[i,2*j] = val_8_30_run1
        matrice_data_8_30[i,(1+2*j)] = val_8_30_run2
        
columns=["pendule_1","pendule_2","main_1","main_2","mainI_1","mainI_2"]
pd.DataFrame(matrice_data_12_15,columns=columns).to_csv("matrice_data_12_15.csv")
pd.DataFrame(matrice_data_8_30,columns=columns).to_csv("matrice_data_8_30.csv")



