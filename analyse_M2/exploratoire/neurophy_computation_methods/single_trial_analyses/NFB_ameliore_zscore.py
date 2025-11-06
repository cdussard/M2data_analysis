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
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)


liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

num_sujet = 19
sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
sample_data_dir = pathlib.Path(sample_data_loc)
date_nom_fichier = dates[num_sujet][-4:]+"-"+dates[num_sujet][3:5]+"-"+dates[num_sujet][0:2]
raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + "_7-2-b.vhdr")

raw_signal = mne.io.read_raw_brainvision(raw_path_sample,preload=False,eog=('HEOG', 'VEOG'))


EpochDataMain = load_data_postICA_postdropBad_windows(liste_rawPathMain,"",True) 

EpochDataPendule = load_data_postICA_postdropBad_windows(liste_rawPathPendule,"",True) 

EpochDataMainIllusion = load_data_postICA_postdropBad_windows(liste_rawPathMainIllusion,"",True) 

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
EpochData = EpochDataMain
EpochData = EpochDataMainIllusion

for epochs_sujet in EpochData:
    print("========================\nsujet"+str(i))
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...")
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,average=False,return_itc=False)#AVERAGE = FALSE
    liste_power_sujets.append(power_sujet)
    i += 1
    
save_tfr_data(liste_power_sujets,liste_rawPathMainIllusion,"alltrials")
save_tfr_data(liste_power_sujets,liste_rawPathMain,"alltrials")#A LANCER APRES

save_tfr_data(liste_power_sujets,liste_rawPathPendule,"alltrials")#on peut pas le copier, ça prend tte la place


#liste_tfr = []
minSubset = 13 #faire une fonction qui fasse les save automatiquement en deux fois pour ne pas saturer ta memoire
maxSubset = 23
listePath = liste_rawPathMainIllusion
liste_tfr = load_tfr_data(listePath[minSubset:maxSubset],"alltrials")
#load_tfr_data(rawPath_main_sujets)
#load_tfr_data(rawPath_pendule_sujets)


dureePreBaseline = 3 #3
dureePreBaseline = - dureePreBaseline
dureeBaseline = 2.0 #2.0
valeurPostBaseline = dureePreBaseline + dureeBaseline

baseline = (dureePreBaseline, valeurPostBaseline)
for tfr in liste_tfr:
    tfr.apply_baseline(baseline=baseline, mode='zscore', verbose=None)
    
i = 0
for i in range(len(liste_tfr)):
    print("averaging "+str(i))
    tfr = liste_tfr[i]
    liste_tfr[i] = tfr.average()#on recupere un objet averageTFR au lieu du epochsTFR
    
#on sauve le tfr averagé par sujet
save_tfr_data(liste_tfr,listePath[minSubset:maxSubset],"avTrial_zscore")

# on fait la grande moyenne

listeTfrAv_main = load_tfr_data(liste_rawPathMain,"avTrial_zscore")
listeTfrAv_mainIllusion = load_tfr_data(liste_rawPathMainIllusion,"avTrial_zscore")
listeTfrAv_pendule = load_tfr_data(liste_rawPathPendule,"avTrial_zscore")

av_power_main_zscore = mne.grand_average(listeTfrAv_main,interpolate_bads=True)
av_power_main_zscore.save("../AV_TFR/all_sujets/main_zscoreIndiv-tfr.h5",overwrite=True)

av_power_mainIllusion_zscore = mne.grand_average(listeTfrAv_mainIllusion,interpolate_bads=True)
av_power_mainIllusion_zscore.save("../AV_TFR/all_sujets/mainIllusion_zscoreIndiv-tfr.h5",overwrite=True)

av_power_pendule_zscore = mne.grand_average(listeTfrAv_pendule,interpolate_bads=True)
av_power_pendule_zscore.save("../AV_TFR/all_sujets/pendule_zscoreIndiv-tfr.h5",overwrite=True)

#========================================
av_power_main_zscore =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main_zscoreIndiv-tfr.h5")[0]
av_power_mainIllusion_zscore =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion_zscoreIndiv-tfr.h5")[0]

av_power_pendule_zscore =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule_zscoreIndiv-tfr.h5")[0]

#cartos
v = 0.25
av_power_pendule_zscore.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-v,vmax=v,cmap=my_cmap)
av_power_main_zscore.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-v,vmax=v,cmap=my_cmap)
av_power_mainIllusion_zscore.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-v,vmax=v,cmap=my_cmap)
raw_signal.plot(block=True)

av_power_pendule_zscore.plot(picks="C3",fmax=40,vmin=-v,vmax=v)
av_power_main_zscore.plot(picks="C3",fmax=40,vmin=-v,vmax=v)
av_power_mainIllusion_zscore.plot(picks="C3",fmax=40,vmin=-v,vmax=v)
raw_signal.plot(block=True)


#====================== zscore GLOBAL================================
liste_tfr_main = load_tfr_data(liste_rawPathMain,"")
liste_tfr_pendule = load_tfr_data(liste_rawPathPendule,"")
liste_tfr_mainIllusion = load_tfr_data(liste_rawPathMainIllusion,"")

baseline = (-3, -1)
for tfr in liste_tfr_mainIllusion:
    tfr.apply_baseline(baseline=baseline, mode='zscore', verbose=None)#les data indiv ne sont pas sauvees

av_power_main_zscoreGlobal = mne.grand_average(liste_tfr_main,interpolate_bads=True)
av_power_pendule_zscoreGlobal = mne.grand_average(liste_tfr_pendule,interpolate_bads=True)
av_power_mainIllusion_zscoreGlobal = mne.grand_average(liste_tfr_mainIllusion,interpolate_bads=True)

# av_power_pendule_zscoreGlobal.save("../AV_TFR/all_sujets/pendule_zscoreGlobal-tfr.h5",overwrite=True)
# av_power_main_zscoreGlobal.save("../AV_TFR/all_sujets/main_zscoreGlobal-tfr.h5",overwrite=True)
# av_power_mainIllusion_zscoreGlobal.save("../AV_TFR/all_sujets/mainIllusion_zscoreGlobal-tfr.h5",overwrite=True)

av_power_main_zscoreGlobal =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main_zscoreGlobal-tfr.h5")[0]
av_power_mainIllusion_zscoreGlobal =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion_zscoreGlobal-tfr.h5")[0]
av_power_pendule_zscoreGlobal =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule_zscoreGlobal-tfr.h5")[0]

v = 0.7
av_power_pendule_zscoreGlobal.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-v,vmax=v,cmap=my_cmap)
av_power_main_zscoreGlobal.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-v,vmax=v,cmap=my_cmap)
av_power_mainIllusion_zscoreGlobal.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-v,vmax=v,cmap=my_cmap)
raw_signal.plot(block=True)


v = 1.3
av_power_pendule_zscoreGlobal.plot(picks="C3",fmax=40,vmin=-v,vmax=v)
av_power_main_zscoreGlobal.plot(picks="C3",fmax=40,vmin=-v,vmax=v)
av_power_mainIllusion_zscoreGlobal.plot(picks="C3",fmax=40,vmin=-v,vmax=v)
raw_signal.plot(block=True)

#============== logRatio global =============================
av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0]
av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0]
av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]

v = 0.2
my_cmap = discrete_cmap(13, 'RdBu_r')
av_power_pendule.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-v,vmax=v,cmap=my_cmap)
av_power_main.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-v,vmax=v,cmap=my_cmap)
av_power_mainIllusion.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-v,vmax=v,cmap=my_cmap)
raw_signal.plot(block=True)

#========= cartes de difference================================
avpower_main_moins_pendule = av_power_main - av_power_pendule
avpower_main_moins_mainIllusion = av_power_main - av_power_mainIllusion

avpower_main_moins_pendule_zscoreIndiv = av_power_main_zscore - av_power_pendule_zscore
avpower_main_moins_mainIllusion_zscoreIndiv  = av_power_main_zscore - av_power_mainIllusion_zscore

avpower_main_moins_pendule_zscoreGlobal = av_power_main_zscoreGlobal - av_power_pendule_zscoreGlobal
avpower_main_moins_mainIllusion_zscoreGlobal = av_power_main_zscoreGlobal - av_power_mainIllusion_zscoreGlobal

avpower_main_moins_pendule_zscoreIndiv.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-0.1,vmax=0.1,cmap=my_cmap)
avpower_main_moins_mainIllusion_zscoreIndiv.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-0.1,vmax=0.1,cmap=my_cmap)

avpower_main_moins_mainIllusion.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-0.05,vmax=0.05,cmap=my_cmap)
avpower_main_moins_pendule.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-0.05,vmax=0.05,cmap=my_cmap)


avpower_main_moins_pendule_zscoreGlobal.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-0.25,vmax=0.25,cmap=my_cmap)
avpower_main_moins_mainIllusion_zscoreGlobal.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25,vmin=-0.25,vmax=0.25,cmap=my_cmap)
raw_signal.plot(block=True)


#================================decours temporels===============================
#from functions.frequencyPower_displays_vRevueMovingAverage import *#avec moving average
from functions.frequencyPower_displays import *#sans mA (refaire la fonct avec bool pour faire les 2)
#zscore single trial
plot_allfreqBand_groupByFrequency(av_power_main_zscore,av_power_mainIllusion_zscore,av_power_pendule_zscore,-0.35,0.9)
raw_signal.plot(block=True)

plot_allfreqBand_groupByCondition(av_power_main_zscore,av_power_mainIllusion_zscore,av_power_pendule_zscore,-0.3,0.9)
raw_signal.plot(block=True)

#========zscore average trial=========

plot_allfreqBand_groupByFrequency(av_power_main_zscoreGlobal,av_power_mainIllusion_zscoreGlobal,av_power_pendule_zscoreGlobal,-1.6,1.2)
plot_allfreqBand_groupByCondition(av_power_main_zscoreGlobal,av_power_mainIllusion_zscoreGlobal,av_power_pendule_zscoreGlobal,-1.6,1.2)
raw_signal.plot(block=True)
#=====sans la moving average=============
from functions.frequencyPower_displays_vRevueMovingAverage import *
plot_allfreqBand_groupByFrequency(av_power_main_zscore,av_power_mainIllusion_zscore,av_power_pendule_zscore,-0.3,0.5)
raw_signal.plot(block=True)
# plot_allfreqBand_groupByCondition(av_power_main_zscore,av_power_mainIllusion_zscore,av_power_pendule_zscore,-0.3,0.5)
# raw_signal.plot(block=True)
plot_allfreqBand_groupByFrequency(av_power_main_zscoreGlobal,av_power_mainIllusion_zscoreGlobal,av_power_pendule_zscoreGlobal,-1.4,0.8)
raw_signal.plot(block=True)

