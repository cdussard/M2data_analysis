# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:54:39 2023

@author: claire.dussard
"""

import os 
import seaborn as sns
import pathlib
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *
import numpy as np 
import mne

nom_essai = "4"
essaisFeedbackSeul = ["pas_enregistre","sujet jeté",
"4","4","sujet jeté","4","4","4","4","MISSING","4","4",
"4","4","4","4","4","4","4","4-b","4","4","4","4","4","4"]

# essaisFeedbackSeul = [nom_essai for i in range(25)]
my_cmap = discrete_cmap(13, 'RdBu')
my_cmap_rev = my_cmap.reversed()

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()
sujetsPb = [0,9]
for sujetpb in sujetsPb:
    allSujetsDispo.remove(sujetpb)
liste_rawPathEffetFBseul = createListeCheminsSignaux(essaisFeedbackSeul,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
# plot les NFB a cote des rest
liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)


liste_tfr_p = load_tfr_data_windows(liste_rawPathEffetFBseul,"pendule",True)
liste_tfr_m = load_tfr_data_windows(liste_rawPathEffetFBseul,"main",True)
liste_tfr_mi = load_tfr_data_windows(liste_rawPathEffetFBseul,"mainIllusion",True)




#do the baseline
dureePreBaseline = 3.4
dureePreBaseline = - dureePreBaseline
dureeBaseline = 1.
valeurPostBaseline = dureePreBaseline + dureeBaseline
baseline = (dureePreBaseline, valeurPostBaseline)

#checker la tete des baseline vite fait
fig, axs = plt.subplots(1, 7)
for (tfr,i) in zip(liste_tfr_p[0:],range(7)):
    tfr.plot(picks="C3",vmin=0,vmax=4e-10,axes=axs[i],cmap="Reds",tmin=-4,tmax=1,fmax=40)
    
fig, axs = plt.subplots(1, 7)
for (tfr,i) in zip(liste_tfr_m[18:],range(7)):
    tfr.plot(picks="C3",vmin=0,vmax=4e-10,axes=axs[i],cmap="Reds",tmin=-4,tmax=1,fmax=40)
    

for tfr in liste_tfr_p:
    tfr.plot(picks="C3",vmin=-0.4,vmax=0.4)

for tfr_p,tfr_m,tfr_mi in zip(liste_tfr_p,liste_tfr_m,liste_tfr_mi):
    tfr_p.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_m.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_mi.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    

# liste_tfr_p.pop(10)
# liste_tfr_m.pop(14)
# liste_tfr_p.pop(13)
# liste_tfr_m.pop(9)
# liste_tfr_m.pop(8)
# liste_tfr_m.pop(5)


for tfr_p in liste_tfr_p:
    tfr_p.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    
for tfr_m in liste_tfr_m:
    tfr_m.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    
for tfr_mi in liste_tfr_mi:
    tfr_mi.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    
fig, axs = plt.subplots(1, 7)
for (tfr,i) in zip(liste_tfr_m[7:],range(7)):
    tfr.plot(picks="C3",vmin=-0.4,vmax=0.4,axes=axs[i],fmax=40)
    
fig, axs = plt.subplots(1, 7)
for (tfr,i) in zip(liste_tfr_m[0:],range(7)):
    tfr.plot_topomap(vmin=-0.6,vmax=0.43,fmax=30,fmin=8,tmin = 2.5,tmax=26.5,cmap=my_cmap_rev,axes=axs[i],colorbar=False)


    
#get grand average
av_power_NF_p = mne.grand_average(liste_tfr_p,interpolate_bads=True)
av_power_NF_p.plot(picks="C3",vmin=-0.4,vmax=0.4,fmax=40)
av_power_NF_p.plot_topomap(vmin=-0.26,vmax=0.26,fmax=30,fmin=8,tmin = 2.5,tmax=26.5,cmap=my_cmap_rev)

#av_power_NF_p.save("../AV_TFR/all_sujets/effet_pendule_logratio-tfr.h5",overwrite=True)

av_power_NF_m = mne.grand_average(liste_tfr_m,interpolate_bads=True)

av_power_NF_m.plot(picks="C3",vmin=0,vmax=4e-10,fmax=40)
av_power_NF_m.plot(picks="C3",vmin=-0.4,vmax=0.4,fmax=40)
av_power_NF_m.plot_topomap(vmin=-0.26,vmax=0.26,fmax=30,fmin=8,tmin = 2.5,tmax=26.5,cmap=my_cmap_rev)


av_power_NF_m.plot_topomap(vmin=0,vmax=3e-10,fmax=30,fmin=8,tmin = 2.5,tmax=26.5,cmap="Reds")
av_power_NF_m.plot_topomap(vmin=0,vmax=3e-10,fmax=30,fmin=8,tmin = -3,tmax=-1,cmap="Reds")
#av_power_NF_m.save("../AV_TFR/all_sujets/effet_main_noBaseline-tfr.h5",overwrite=True)
#av_power_NF_m.save("../AV_TFR/all_sujets/effet_main_logratio-tfr.h5",overwrite=True)


av_power_NF_mi = mne.grand_average(liste_tfr_mi,interpolate_bads=True)
av_power_NF_mi.plot(picks="C3",vmin=-0.4,vmax=0.4,fmax=40)
av_power_NF_mi.plot_topomap(vmin=-0.26,vmax=0.26,fmax=30,fmin=8,tmin = 2.5,tmax=26.5,cmap=my_cmap_rev)
#av_power_NF_mi.save("../AV_TFR/all_sujets/effet_mainIll_logratio-tfr.h5",overwrite=True)


av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/effet_main_logratio-tfr.h5")[0]
av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/effet_main_noBaseline-tfr.h5")[0]
av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/effet_mainIll_logratio-tfr.h5")[0]
av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/effet_pendule_logratio-tfr.h5")[0]

av_power_pendule.plot_topomap(vmin=-0.43,vmax=0.43,fmax=30,fmin=8,tmin = 2.5,tmax=26.5,cmap=my_cmap_rev)
av_power_main.plot_topomap(vmin=-0.43,vmax=0.43,fmax=30,fmin=8,tmin = 2.5,tmax=26.5,cmap=my_cmap_rev)
av_power_mainIllusion.plot_topomap(vmin=-0.43,vmax=0.43,fmax=30,fmin=8,tmin = 2.5,tmax=26.5,cmap=my_cmap_rev)


av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main_noBaseline-tfr.h5")[0]
av_power_main.plot(picks="C3",vmin=0,vmax=4e-10,fmax=40)
av_power_main.plot_topomap(vmin=0,vmax=3e-10,fmax=30,fmin=8,tmin = 2.5,tmax=26.5,cmap="Reds")
av_power_main.plot_topomap(vmin=0,vmax=3e-10,fmax=30,fmin=8,tmin = -3,tmax=-1,cmap="Reds")

av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0]
av_power_main.plot_topomap(vmin=-0.26,vmax=0.26,fmax=30,fmin=8,tmin = 2.5,tmax=26.5,cmap=my_cmap_rev)