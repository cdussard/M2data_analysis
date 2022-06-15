# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:43:12 2022

@author: claire.dussard
"""

from functions.load_savedData import *
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.frequencyPower_displays import *
import numpy as np
import os
import pandas as pd

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

#load le sujet pour plot block = True
num_sujet = 19
sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
sample_data_dir = pathlib.Path(sample_data_loc)
date_nom_fichier = dates[num_sujet][-4:]+"-"+dates[num_sujet][3:5]+"-"+dates[num_sujet][0:2]
raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + "_7-2-b.vhdr")

raw_signal = mne.io.read_raw_brainvision(raw_path_sample,preload=False,eog=('HEOG', 'VEOG'))

liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule[15:],"",True)
liste_tfrMain = load_tfr_data_windows(liste_rawPathMain[15:],"",True)
liste_tfrMainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion[15:],"",True)

my_cmap = discrete_cmap(13, 'RdBu_r')

liste_tfrPendule[0].plot_topomap(tmin=1.5,tmax=26.5,fmin=8,fmax=30,baseline=(-3,-1),mode="logratio",cmap=my_cmap)

raw_signal.plot(block=True)

for i in range(8):
    liste_tfrPendule[i].plot(picks="C3",fmin=3,fmax=40,baseline=(-3,-1),mode="logratio",vmin=-1,vmax=1)
    liste_tfrMain[i].plot(picks="C3",fmin=3,fmax=40,baseline=(-3,-1),mode="logratio",vmin=-1,vmax=1)
    liste_tfrMainIllusion[i].plot(picks="C3",fmin=3,fmax=40,baseline=(-3,-1),mode="logratio",vmin=-1,vmax=1)
    raw_signal.plot(block=True)


#======== moyenner les donnees par sujet ================
#test pour sujet 00 
tfrPendule_s0 = liste_tfrPendule[0]
tfrMain_s0  = liste_tfrMain[0]
tfrMainIllusion_s0  = liste_tfrMainIllusion[0] 

tfrPendule_s0.apply_baseline(baseline = (-3,-1),mode="logratio")
tfrMain_s0.apply_baseline(baseline = (-3,-1),mode="logratio")
tfrMainIllusion_s0.apply_baseline(baseline = (-3,-1),mode="logratio")

tfrPendule_s0.plot(picks="C3")
tfrMain_s0.plot(picks="C3")
tfrMainIllusion_s0.plot(picks="C3")

moy_s00 = (tfrPendule_s0 + tfrMain_s0 + tfrMainIllusion_s0)/3

moy_s00 = (tfrPendule_s0 + tfrMain_s0 + tfrMainIllusion_s0)/3

moy_s00.plot(picks="C3")
raw_signal.plot(block=True)

#pour moyenner temps et frequences (dim = 1 et 2)
moy_FreqTpsEcrase = np.mean(moy_s00.data,axis=(1,2))
