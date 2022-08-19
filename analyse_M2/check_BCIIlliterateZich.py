# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:09:25 2022

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

liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule,"",True)
liste_tfrMain = load_tfr_data_windows(liste_rawPathMain,"",True)
liste_tfrMainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion,"",True)


# plot power courbe during baseline VS during trial 
#Real-time EEG feedback during simultaneous EEG–fMRI identifies the cortical signature of motor imagery Zich et al 2015
#liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)

dureePreBaseline = 3 #3
dureePreBaseline = - dureePreBaseline
dureeBaseline = 2.0 #2.0
valeurPostBaseline = dureePreBaseline + dureeBaseline

baseline = (dureePreBaseline, valeurPostBaseline)
for tfr_p,tfr_m,tfr_mi in zip(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion):
    tfr_p.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_m.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_mi.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    

def printC3C4_cond_participant(liste_tfr,num_sujet ,fmin,fmax,fstep,ax):
    data = liste_tfr[num_sujet].data
    data_meanTps = np.mean(data,axis=2)
    data_C3 = data_meanTps[11][fmin:fmax]
    data_C4 = data_meanTps[13][fmin:fmax]
    freqs = np.arange(fmin, fmax,fstep)
    
    ax.plot(freqs,data_C3, label='C3') #apres baseline
    ax.plot(freqs,data_C4,label="C4")
    ax.axvline(x=8,color="black",linestyle="--")
    ax.axvline(x=30,color="black",linestyle="--")
    ax.legend(loc="upper left")
    ax.set_ylim(top=0.3,bottom=-0.6)

def printC3C4_cond(liste_tfr,fmin,fmax,fstep):
    
    fig, axs = plt.subplots(4,6)
    ind_sujet = 0  
    for ligne in range(4):
        for colonne in range(6):
            print(ind_sujet)
            ax = axs[ligne,colonne]
            printC3C4_cond_participant(liste_tfr,ind_sujet,fmin,fmax,fstep,ax)
            if ind_sujet==22:
                break
            ind_sujet += 1 
    

printC3C4_cond(liste_tfrPendule,3,40,1)
printC3C4_cond(liste_tfrMain,3,40,1)
printC3C4_cond(liste_tfrMainIllusion,3,40,1)
raw_signal.plot(block=True)

def print_allConds_participant(num_sujet ,fmin,fmax,fstep,ax):
    printC3C4_cond_participant(liste_tfrPendule,num_sujet ,fmin,fmax,fstep,ax[0])
    printC3C4_cond_participant(liste_tfrMain,num_sujet ,fmin,fmax,fstep,ax[1])
    printC3C4_cond_participant(liste_tfrMainIllusion,num_sujet ,fmin,fmax,fstep,ax[2])
    
n_sujets = 6
round_suj = 4
for r in range(0,round_suj):
    print("r"+str(r))
    fig,axs = plt.subplots(n_sujets,3)
    j = 0
    for i in range(n_sujets*r,n_sujets*(r+1)):
        if i == 23:
            break
        else:
            print_allConds_participant(i,3,40,1,axs[j])
            j += 1
raw_signal.plot(block=True)
