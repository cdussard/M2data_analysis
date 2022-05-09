# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:22:01 2022

@author: claire.dussard
"""
import pandas as pd
import mne
douzeQuinzeHzData = pd.read_csv("./data/Jasp_anova/ANOVA_12_15Hz_C3_long.csv")
df_douzeQuinzeHzData=douzeQuinzeHzData.iloc[: , 1:]
df_douzeQuinzeHzData_mP = df_douzeQuinzeHzData.iloc[:,0:2]
res=mne.stats.permutation_t_test(df_douzeQuinzeHzData_mP, n_permutations=10000, tail=0)
res2=mne.stats.permutation_t_test(df_douzeQuinzeHzData_mP, n_permutations=1000, tail=0)

df_douzeQuinzeHzData_mMi = df_douzeQuinzeHzData.iloc[:,1:3]
mne.stats.permutation_t_test(df_douzeQuinzeHzData_mMi, n_permutations=10000, tail=0)

#difference main vs pendule
df_douzeQuinzeHzData_mP = df_douzeQuinzeHzData.iloc[:,0]-df_douzeQuinzeHzData.iloc[:,0]

#dans l'ideal il faudrait avoir un permutation F test,
# a voir si on peut recuperer l'implementation MNE et l'etendre au F
#on veut faire la meme chose entre 8 et 30 Hz
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

liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule,"",True)
liste_tfrMain = load_tfr_data_windows(liste_rawPathMain,"",True)
liste_tfrMainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion,"",True)


def copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion):
#avoid having to reload from scratch after every ANOVA (instances modified in place)
    liste_tfr_pendule = []
    liste_tfr_main = []
    liste_tfr_mainIllusion = []
    for tfr_p,tfr_m,tfr_mi in zip(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion):
        liste_tfr_pendule.append(tfr_p.copy())
        liste_tfr_main.append(tfr_m.copy())
        liste_tfr_mainIllusion.append(tfr_mi.copy())
    return liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion

#main vs pendule
def data_freq_tTest_perm(elec,fmin,fmax,tmin,tmax,liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule):
    mode_baseline = 'logratio'
    baseline = (-3,-1)
    #compute baseline (first because after we crop time)
    for tfr_m,tfr_mi,tfr_p in zip(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule):
        tfr_m.apply_baseline(baseline=baseline, mode=mode_baseline, verbose=None)
        tfr_mi.apply_baseline(baseline=baseline, mode=mode_baseline, verbose=None)
        tfr_p.apply_baseline(baseline=baseline, mode=mode_baseline, verbose=None)
    #crop time & frequency
    for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
        tfr_mainI.crop(tmin = tmin,tmax=tmax,fmin = fmin,fmax = fmax)
        tfr_main.crop(tmin = tmin,tmax=tmax,fmin = fmin,fmax = fmax)
        tfr_pendule.crop(tmin = tmin,tmax=tmax,fmin = fmin,fmax = fmax)
    #subset electrode
    for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
        tfr_mainI.pick_channels([elec])
        tfr_main.pick_channels([elec])
        tfr_pendule.pick_channels([elec])
    #create ANOVA table "faire evoluer pour plusieurs elecs a la fois
    tableau_mainPendule = np.zeros(shape=(23,23))
    tableau_mainMainIllusion = np.zeros(shape=(23,23))
    for i in range(23):#sujets
        print("sujet"+str(i))
        #ecraser forme electrodes
        liste_tfr_pendule[i].data = np.mean(liste_tfr_pendule[i].data,axis=0)
        liste_tfr_main[i].data = np.mean(liste_tfr_main[i].data,axis=0)
        liste_tfr_mainIllusion[i].data = np.mean(liste_tfr_mainIllusion[i].data,axis=0)
        powerFreq_pendule = np.median(liste_tfr_pendule[i].data,axis=1)#verifier
        #pool time
        powerFreq_main = np.median(liste_tfr_main[i].data,axis=1)
        powerFreq_mainI = np.median(liste_tfr_mainIllusion[i].data,axis=1)
        print(powerFreq_main)
        mainMoinsPendule_i = powerFreq_main - powerFreq_pendule
        print("main moins pendule")
        print(mainMoinsPendule_i)
        mainMoinsMainIllusion_i = powerFreq_main - powerFreq_mainI
        for j in range(fmax-fmin+1):#freq
            print("freq"+str(fmin+j))
            tableau_mainPendule[i][j] = mainMoinsPendule_i[j]
            print(mainMoinsPendule_i[j])
            tableau_mainMainIllusion[i][j] = mainMoinsMainIllusion_i[j]
    return tableau_mainPendule,tableau_mainMainIllusion
   
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
tableau_mainPendule,tableau_mainMainIllusion = data_freq_tTest_perm("C3",8,30,2.5,26.8,liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule)
#t test avec permutation

res = mne.stats.permutation_t_test(tableau_mainPendule,1000000)

res[0]
pval = res[1]
res[2]

for p in pval:
    print(p)
    
res2 = mne.stats.permutation_t_test(tableau_mainMainIllusion,1000000)
pval2 = res2[1]
i = 8
for p in pval2:
    print("freq"+str(i)+"Hz : "+str(p))
    i += 1
