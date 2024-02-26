# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 20:36:55 2024

@author: claire.dussard
"""

import os 
import seaborn as sns
import pathlib
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *
import numpy as np 

#create liste of file paths
essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData() 

nom_essai = "2-2"

allSujetsDispo_MI = allSujetsDispo[2:]

my_cmap = discrete_cmap(13, 'RdBu')
my_cmap_rev = my_cmap.reversed()

#on commence au 3 (reste pbs noms)
liste_rawPath_rawMIalone = []
for num_sujet in allSujetsDispo_MI:
    print("sujet n° "+str(num_sujet))
    nom_sujet = listeNumSujetsFinale[num_sujet]
    if num_sujet in SujetsPbNomFichiers:
        if num_sujet>0:
            date_sujet = '19-04-2021'
        else:
            date_sujet = '15-04-2021'
    else:
        date_sujet = dates[num_sujet]
    date_nom_fichier = date_sujet[-4:]+"-"+date_sujet[3:5]+"-"+date_sujet[0:2]+"_"
    dateSession = listeDatesFinale[num_sujet]
    sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
    sample_data_dir = pathlib.Path(sample_data_loc)
    raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + nom_essai+".vhdr")#il faudrait recup les epochs et les grouper ?
    liste_rawPath_rawMIalone.append(raw_path_sample)

print(liste_rawPath_rawMIalone)

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

liste_power_sujets = load_tfr_data_windows(liste_rawPath_rawMIalone,"",True)
#checker les bl
fig, axs = plt.subplots(1, 7)
for (tfr,i) in zip(liste_power_sujets[13:],range(7)):
    tfr.plot(picks="C3",vmin=0,vmax=4e-10,axes=axs[i],cmap="Reds",tmin=-4,tmax=1,fmax=40)
    
#do the baseline
dureePreBaseline = 4.0 
dureePreBaseline = - dureePreBaseline
dureeBaseline = 2
valeurPostBaseline = dureePreBaseline + dureeBaseline
baseline = (dureePreBaseline, valeurPostBaseline)

for tfr in liste_power_sujets:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)

fig, axs = plt.subplots(1, 7)
for (tfr,i) in zip(liste_power_sujets[0:],range(7)):
    tfr.plot(picks="C3",vmin=-0.4,vmax=0.4,axes=axs[i],fmax=40)
    
fig, axs = plt.subplots(1, 7)
for (tfr,i) in zip(liste_power_sujets[0:],range(7)):
    tfr.plot_topomap(vmin=-0.4,vmax=0.4,fmax=30,fmin=30,tmin = 2.5,tmax=26.5,cmap=my_cmap_rev,axes=axs[i],colorbar=False)
    
av_power_NF_mialone = mne.grand_average(liste_power_sujets,interpolate_bads=True)
#av_power_NF_mialone.save("../AV_TFR/all_sujets/MIalone_logratio-tfr.h5",overwrite=True)
av_power_NF_mialone =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/MIalone_logratio-tfr.h5")[0]
       

av_power_NF_mialone.plot(picks="C3",vmin=-0.4,vmax=0.4,fmax=40)
av_power_NF_mialone.plot_topomap(vmin=-0.26,vmax=0.26,fmax=30,fmin=8,tmin = 2.5,tmax=26.5,cmap=my_cmap_rev)

