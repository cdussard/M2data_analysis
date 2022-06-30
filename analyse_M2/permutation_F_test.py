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
    tableau_mainPendule = np.zeros(shape=(23,fmax-fmin+1))#23 = nb sujets
    tableau_mainMainIllusion = np.zeros(shape=(23,fmax-fmin+1))
    tableau_main = np.zeros(shape=(23,fmax-fmin+1))
    tableau_pendule = np.zeros(shape=(23,fmax-fmin+1))
    tableau_mainIllusion = np.zeros(shape=(23,fmax-fmin+1))
    for i in range(23):#sujets
        print("sujet"+str(i))
        #ecraser forme electrodes
        liste_tfr_pendule[i].data = np.mean(liste_tfr_pendule[i].data,axis=0)
        liste_tfr_main[i].data = np.mean(liste_tfr_main[i].data,axis=0)
        liste_tfr_mainIllusion[i].data = np.mean(liste_tfr_mainIllusion[i].data,axis=0)
        powerFreq_pendule = np.median(liste_tfr_pendule[i].data,axis=1)#OK donc dim = freq x time
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
            tableau_main[i][j] = powerFreq_main[j]
            tableau_pendule[i][j] = powerFreq_pendule[j]
            tableau_mainIllusion[i][j] = powerFreq_mainI[j]
    return tableau_mainPendule,tableau_mainMainIllusion,tableau_main,tableau_pendule,tableau_mainIllusion
   
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
tableau_mainPendule,tableau_mainMainIllusion,tableau_main,tableau_pendule,tableau_mainIllusion = data_freq_tTest_perm("C3",8,30,2.5,26.8,liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule)
listeSuj = [0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

#y a t'il une desynchro plus forte que zero entre 8 et 30 Hz
T0, p_values_m, H0 = mne.stats.permutation_t_test(tableau_main,1000000)
print(T0)
print(p_values)
print(H0)



T0, p_values_mi, H0  = mne.stats.permutation_t_test(tableau_mainIllusion,1000000)
print(T0)
print(p_values)
print(H0)
significant_freqs = p_values <= 0.05
print(significant_freqs)

T0, p_values_p, H0  = mne.stats.permutation_t_test(tableau_pendule,1000000)
print(T0)
print(p_values)
print(H0)
significant_freqs = p_values <= 0.05
print(significant_freqs)

logp = -log10(p_values_p)

freqValues = range(8,31)
plt.scatter(freqValues,p_values_m,label="main")
plt.scatter(freqValues,p_values_mi,label="main+vib")
plt.scatter(freqValues,p_values_p,label="pendule")
plt.axhline(y=0.05/(23*3),color="black")
plt.legend(loc="upper left")
raw_signal.plot(block=True)

for sujet in range(23):
    print("sujet n°"+str(listeSuj[sujet]))
    mean_12_15 = np.mean(tableau_mainPendule[sujet][4:8])#12-15Hz
    if mean_12_15<0:
        print("NEG")
    else:
        print("POS")
    print(mean_12_15)

#===========meme chose entre 3 et 85 Hz===============

liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
tableau_mainPendule,tableau_mainMainIllusion,tableau_main,tableau_pendule,tableau_mainIllusion = data_freq_tTest_perm("C3",3,84,2.5,26.8,liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule)

T0, p_values_m, H0 = mne.stats.permutation_t_test(tableau_main,1000000)
T0, p_values_p, H0  = mne.stats.permutation_t_test(tableau_pendule,1000000)
T0, p_values_mi, H0  = mne.stats.permutation_t_test(tableau_mainIllusion,1000000)

from itertools import compress
significant_freqs_m = p_values_m <= 0.05
freqSignif_m = list(compress(freqValues, significant_freqs_m))
print(freqSignif_m)

significant_freqs_p = p_values_p <= 0.05
freqSignif_p = list(compress(freqValues, significant_freqs_p))
print(freqSignif_p)

significant_freqs_mi = p_values_mi <= 0.05
freqSignif_mi = list(compress(freqValues, significant_freqs_mi))
print(freqSignif_mi)


log_p_values_m = np.log10(p_values_m)
log_p_values_p = np.log10(p_values_p)
log_p_values_mi = np.log10(p_values_mi)



freqValues = range(3,85,1)
fig,ax = plt.subplots()
plt.plot(freqValues,log_p_values_m,label="main")
plt.plot(freqValues,log_p_values_mi,label="main+vib")
plt.plot(freqValues,log_p_values_p,label="pendule")
plt.axhline(y=np.log10(0.05),color="black")
ax.axvline(x=8,color="black",linestyle="--")
ax.axvline(x=30,color="black",linestyle="--")
plt.legend(loc="upper left")
raw_signal.plot(block=True)


# ===== fin du test 3 - 85 Hz
#t test avec permutation : y a t'il une difference de desynchro main/pendule plus forte que zero
res = mne.stats.permutation_t_test(tableau_mainPendule,100000)

res[0]
pval = res[1]
res[2]

i = 8
for p in pval:
    print("freq"+str(i)+"Hz : "+str(p))
    i += 1
    
res2 = mne.stats.permutation_t_test(tableau_mainMainIllusion,100000)
pval2 = res2[1]
i = 8
for p in pval2:
    print("freq"+str(i)+"Hz : "+str(p))
    i += 1
    
    
#avec clustering
res_c = mne.stats.permutation_cluster_test(tableau_pendule,n_permutations=1000)
print(res_c[0])#F
print(res_c[1])#cluster

print(res_c[2])#cluster p value
print(res_c[3])#H0

