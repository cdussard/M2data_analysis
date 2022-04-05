#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:51:16 2022

@author: claire.dussard
"""

#mne.make_fixed_length_epochs(raw, duration=1.0, preload=False, reject_by_annotation=True, proj=True, overlap=0.0, verbose=None)
import pandas as pd

liste_tfr_main  = load_tfr_data(rawPath_main_sujets,"")
#without baseline
av_power_main_withoutBaseline = mne.grand_average(liste_tfr_main,interpolate_bads=True)

liste_tfr_pendule = load_tfr_data(liste_rawPathPendule,"")
av_power_pendule_withoutBaseline = mne.grand_average(liste_tfr_pendule,interpolate_bads=True)

liste_tfr_mainIllusion = load_tfr_data(liste_rawPathMainIllusion,"")
av_power_mainIllusion_withoutBaseline = mne.grand_average(liste_tfr_mainIllusion,interpolate_bads=True)
#av_power_mainIllusion_withoutBaseline.save("../withoutBaseline/mainIllusion_mean-tfr.hdf5")


for sujet_sansBaseline in liste_tfr:
    #sujet_sansBaseline.plot(picks="C3",fmin=3,fmax=40)
    sujet_sansBaseline.plot_topomap(fmin=8,fmax=30,tmin=1.5,tmax=25.5)
    
av_power_pendule_withoutBaseline.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=26.5,cmap='RdBu_r')
av_power_main_withoutBaseline.plot_topomap(fmin=8,fmax=30,tmin=1.5,tmax=25.5,cmap='RdBu_r')
av_power_mainIllusion_withoutBaseline.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=26.5)#cmap='RdBu_r')
raw_signal.plot(block=True)

av_power_main.plot(picks="C3",fmin=3,fmax=40)



#affichage des TFR sans Baseline
av_power_pendule_withoutBaseline.plot(picks="C3",fmin=3,fmax=40,vmin=0,vmax=2.0e-10)
av_power_main_withoutBaseline.plot(picks="C3",fmin=3,fmax=40,vmin=0)

#===========ADD AXE WITH TIMES WITH VIBRATION==============
fig,axes = plt.subplots()
av_power_mainIllusion_withoutBaseline.plot(picks="C3",fmin=3,fmax=40,vmin=0,vmax=1.8e-10,axes=axes)
axes.axvline(2, color='green', linestyle='--')
axes.axvline(6.5, color='black', linestyle='--')
axes.axvline(8.3, color='black', linestyle='--')
axes.axvline(12.7, color='black', linestyle='--')
axes.axvline(14.5, color='black', linestyle='--')
axes.axvline(18.92, color='black', linestyle='--')
axes.axvline(20.72, color='black', linestyle='--')
axes.axvline(25.22, color='black', linestyle='--')
axes.axvline(27.02, color='black', linestyle='--')
axes.axvline(26.9, color='green', linestyle='--')
plt.show()
raw_signal.plot(block=True)

#with seuils
seuils_sujets = pd.read_csv("./data/seuil_data/seuils_sujets_dash.csv")
for i in range(23):
    seuil = float(seuils_sujets["seuil_min_mvt"][i])
    print(seuil)
    main_seuil = liste_tfr_main[i].data/seuil
    liste_tfr_main[i].data = main_seuil
    pendule_seuil = liste_tfr_pendule[i].data/seuil
    liste_tfr_pendule[i].data = pendule_seuil
    #mainIllusion_seuil = liste_tfr_mainIllusion[i].data/seuil
    #liste_tfr_mainIllusion[i].data = mainIllusion_seuil

av_power_main_noBL_seuil = mne.grand_average(liste_tfr_main,interpolate_bads=True)
av_power_pendule_noBL_seuil = mne.grand_average(liste_tfr_pendule,interpolate_bads=True)
av_power_main_noBL_seuil.save("../withoutBaseline/mainSeuil_mean-tfr.h5")
av_power_pendule_noBL_seuil.save("../withoutBaseline/penduleSeuil_mean-tfr.h5")


# for i in range(23):
#     seuil = float(seuils_sujets["seuil_min_mvt"][i])
#     print(seuil)
#     pendule_seuil = liste_tfr_mainIllusion[i].data/seuil
#     liste_tfr_mainIllusion[i].data = mainIllusion_seuil
# av_power_mainIllusion_noBL_seuil = mne.grand_average(liste_tfr_mainIllusion,interpolate_bads=True)
#av_power_mainIllusion_noBL_seuil.save("../withoutBaseline/mainIllusionSeuil_mean-tfr.hdf5")

fig,axes = plt.subplots()
av_power_mainIllusion_noBL_seuil.plot(picks="C3",fmin=3,fmax=40,vmin=0,vmax=1.0e-11,axes=axes)
axes.axvline(2, color='green', linestyle='--')
axes.axvline(6.5, color='black', linestyle='--')
axes.axvline(8.3, color='black', linestyle='--')
axes.axvline(12.7, color='black', linestyle='--')
axes.axvline(14.5, color='black', linestyle='--')
axes.axvline(18.92, color='black', linestyle='--')
axes.axvline(20.72, color='black', linestyle='--')
axes.axvline(25.22, color='black', linestyle='--')
axes.axvline(27.02, color='black', linestyle='--')
axes.axvline(26.9, color='green', linestyle='--')
plt.show()
fig
raw_signal.plot(block=True)

#avec la baseline
av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0]
fig,axes = plt.subplots()
av_power_mainIllusion.plot(picks="C3",fmin=3,fmax=40,axes=axes,vmin=-0.4,vmax=0.4)#,baseline=(-3,-1))#,vmax=1.0e-11,axes=axes)vmin=0,
axes.axvline(2, color='green', linestyle='--')
axes.axvline(6.5, color='black', linestyle='--')
axes.axvline(8.3, color='black', linestyle='--')
axes.axvline(12.7, color='black', linestyle='--')
axes.axvline(14.5, color='black', linestyle='--')
axes.axvline(18.92, color='black', linestyle='--')
axes.axvline(20.72, color='black', linestyle='--')
axes.axvline(25.22, color='black', linestyle='--')
axes.axvline(27.02, color='black', linestyle='--')
axes.axvline(26.9, color='green', linestyle='--')
plt.show()
fig
raw_signal.plot(block=True)



#====================================================

#cartes de difference

avpower_main_moins_pendule_noBL =   av_power_main_withoutBaseline - av_power_pendule_withoutBaseline

avpower_main_moins_mainIllusion_noBL = av_power_main_withoutBaseline - av_power_mainIllusion_withoutBaseline
#TFR DIFFERENCE
#avpower_main_moins_pendule_noBL.plot_topomap(fmin=8,fmax=30,tmin=1.5,tmax=25.5)
avpower_main_moins_pendule_noBL.plot(picks="C3",fmin=3,fmax=40)
avpower_main_moins_mainIllusion_noBL.plot(picks="C3",fmin=3,fmax=40)
raw_signal.plot(block=True)

#TOPOMAP DIFF
avpower_main_moins_pendule_noBL.plot_topomap(fmin=8,fmax=30,tmin=1.5,tmax=25.5,cmap='RdBu_r')
avpower_main_moins_mainIllusion_noBL.plot_topomap(fmin=8,fmax=30,tmin=1.5,tmax=25.5,cmap='RdBu_r')
raw_signal.plot(block=True)


#cartes avec seuil de diff
av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0]
av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0]
av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]

avpower_main_moins_pendule = av_power_main - av_power_pendule
avpower_main_moins_mainIllusion = av_power_main - av_power_mainIllusion

#TFR DIFFERENCE
avpower_main_moins_pendule.plot(picks="C3",fmin=3,fmax=40)
avpower_main_moins_mainIllusion.plot(picks="C3",fmin=3,fmax=40)
raw_signal.plot(block=True)



 #faire le grand average sans baseline mais avec seuil
liste_tfr = []
liste_tfr_mainIllusion = load_tfr_data(rawPath_mainIllusion_sujets,"")
liste_tfr_pendule = load_tfr_data(rawPath_pendule_sujets,"")
liste_tfr_main = load_tfr_data(rawPath_main_sujets,"")

#load_tfr_data(rawPath_main_sujets)
#load_tfr_data(rawPath_pendule_sujets)

#===================apply SEUIL, not BL by subject before grand averaging=========================
for i in range(23):
    seuil = float(seuils_sujets["seuil_min_mvt"][i])
    print(seuil)
    main_seuil = liste_tfr_main[i].data/seuil
    liste_tfr_main[i].data = main_seuil
    pendule_seuil = liste_tfr_pendule[i].data/seuil
    liste_tfr_pendule[i].data = pendule_seuil
    mainIllusion_seuil = liste_tfr_mainIllusion[i].data/seuil
    liste_tfr_mainIllusion[i].data = mainIllusion_seuil


#================compute grand average===============================================
#get bad sujets out : pop 4 et 12

av_power_main_noBL_seuil = mne.grand_average(liste_tfr_main,interpolate_bads=True)
av_power_mainIllusion_noBL_seuil = mne.grand_average(liste_tfr_mainIllusion,interpolate_bads=True)
av_power_pendule_noBL_seuil = mne.grand_average(liste_tfr_pendule,interpolate_bads=True)

#plotTFR SANS BASELINE MAIS AVEC SEUIL
av_power_main_noBL_seuil.plot(picks="C3",fmin=3,fmax=40,vmin=0)
av_power_mainIllusion_noBL_seuil.plot(picks="C3",fmin=3,fmax=40,vmin=0)
av_power_pendule_noBL_seuil.plot(picks="C3",fmin=3,fmax=40,vmin=0)
raw_signal.plot(block=True)


av_power_main_noBL_seuil.plot(picks="C4",fmin=3,fmax=40,vmin=0)
av_power_mainIllusion_noBL_seuil.plot(picks="C4",fmin=3,fmax=40,vmin=0)
av_power_pendule_noBL_seuil.plot(picks="C4",fmin=3,fmax=40,vmin=0)
raw_signal.plot(block=True)


#CARTE TOPOMAP
av_power_main_noBL_seuil.plot_topomap(fmin=8,fmax=30,tmin=1.5,tmax=25.5,cmap='RdBu_r')
av_power_mainIllusion_noBL_seuil.plot_topomap(fmin=8,fmax=30,tmin=1.5,tmax=25.5,cmap='RdBu_r')
av_power_pendule_noBL_seuil.plot_topomap(fmin=8,fmax=30,tmin=1.5,tmax=25.5,cmap='RdBu_r')
raw_signal.plot(block=True)

#DIFFERENCES

avpower_main_moins_pendule_noBLseuil = av_power_main_noBL_seuil - av_power_pendule_noBL_seuil
avpower_main_moins_mainIllusion_noBLseuil = av_power_main_noBL_seuil - av_power_mainIllusion_noBL_seuil
