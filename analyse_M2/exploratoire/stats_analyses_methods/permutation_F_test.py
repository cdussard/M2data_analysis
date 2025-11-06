# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:22:01 2022

@author: claire.dussard
"""
import mne

#dans l'ideal il faudrait avoir un permutation F test,
# a voir si on peut recuperer l'implementation MNE et l'etendre au F
#on veut faire la meme chose entre 8 et 30 Hz
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

liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule[0:2],"",True)
liste_tfrMain = load_tfr_data_windows(liste_rawPathMain[0:2],"",True)
liste_tfrMainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion[0:2],"",True)


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
    n_sujets = len(liste_tfr_pendule)
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
    tableau_mainPendule = np.zeros(shape=(n_sujets,fmax-fmin+1))#23 = nb sujets
    tableau_mainMainIllusion = np.zeros(shape=(n_sujets,fmax-fmin+1))
    tableau_main = np.zeros(shape=(n_sujets,fmax-fmin+1))
    tableau_pendule = np.zeros(shape=(n_sujets,fmax-fmin+1))
    tableau_mainIllusion = np.zeros(shape=(n_sujets,fmax-fmin+1))
    for i in range(n_sujets):#sujets
        print("sujet"+str(i))
        #ecraser forme electrodes
        liste_tfr_pendule[i].data = np.mean(liste_tfr_pendule[i].data,axis=0)
        liste_tfr_main[i].data = np.mean(liste_tfr_main[i].data,axis=0)
        liste_tfr_mainIllusion[i].data = np.mean(liste_tfr_mainIllusion[i].data,axis=0)
        #pool time
        powerFreq_pendule = np.median(liste_tfr_pendule[i].data,axis=1)#OK donc dim = freq x time
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
plt.hist(H0)
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


def print_pval_permAgainstZero(elec):
    liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
    tableau_mainPendule,tableau_mainMainIllusion,tableau_main,tableau_pendule,tableau_mainIllusion = data_freq_tTest_perm(elec,3,84,2.5,26.8,liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule)
    freqValues = range(3,85,1)
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
    
    
    fig,ax = plt.subplots()
    plt.plot(freqValues,log_p_values_p,label="pendule")
    plt.plot(freqValues,log_p_values_m,label="main")
    plt.plot(freqValues,log_p_values_mi,label="main+vib")
    plt.axhline(y=np.log10(0.05),color="black")
    ax.axvline(x=8,color="black",linestyle="--")
    ax.axvline(x=30,color="black",linestyle="--")
    plt.legend(loc="upper left")
    raw_signal.plot(block=True)
    return p_values_p,p_values_m,p_values_mi
    


av_power_main.plot(picks="C3",vmin = -0.26,vmax = 0.26)
av_power_mainIllusion.plot(picks="C3",vmin = -0.26,vmax = 0.26)
av_power_pendule .plot(picks="C3",vmin = -0.26,vmax = 0.26)
raw_signal.plot(block=True)

# ===== fin du test 3 - 85 Hz========
#faire le t-test 3-85Hz sur les 28 electrodes #
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
tableau_mainPendule,tableau_mainMainIllusion,tableau_main,tableau_pendule,tableau_mainIllusion = data_freq_tTest_perm("C3",3,84,2.5,26.8,liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule)

p_values_p,p_values_m,p_values_mi = print_pval_permAgainstZero("C4")

av_power_pendule .plot(picks="C4",vmin = -0.26,vmax = 0.26)
av_power_main.plot(picks="C4",vmin = -0.26,vmax = 0.26)
av_power_mainIllusion.plot(picks="C4",vmin = -0.26,vmax = 0.26)
raw_signal.plot(block=True)

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

# en Cz
p_values_p,p_values_m,p_values_mi = print_pval_permAgainstZero("Cz")

av_power_pendule .plot(picks="Cz",vmin = -0.26,vmax = 0.26)
av_power_main.plot(picks="Cz",vmin = -0.26,vmax = 0.26)
av_power_mainIllusion.plot(picks="Cz",vmin = -0.26,vmax = 0.26)
raw_signal.plot(block=True)


#nouvel essai pas opti du tout MAIS QUI MARCHE
av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]
liste_pendule = []
liste_main = []
liste_mainIllusion = []
#
for elec in av_power_pendule.ch_names:
    print("ELEC  "+elec)
    liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)

    tableau_mainPendule,tableau_mainMainIllusion,tableau_main,tableau_pendule,tableau_mainIllusion = data_freq_tTest_perm(elec,3,84,2.5,25.5,liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule)
    liste_mainIllusion.append(tableau_mainIllusion)
    liste_main.append(tableau_main)
    liste_pendule.append(tableau_pendule)
    
#ICI CA MARCHE
def get_pvalue_allElec_allFreq(liste_condition,npermut):
    n_sujets = liste_condition[0].shape[0]
    liste_suj_data = []
    for suj in range(n_sujets):
        for elec in range(28):
            print("suj"+str(suj))
            liste_suj_data.append(liste_condition[elec][suj])
    
    fullTableTest_condition = np.zeros(shape=(n_sujets,82*28))
    for i in range(n_sujets):#sujet
        for j in range(28):#electrodes
            fullTableTest_condition[i:i+1,j*82:(j+1)*82] = liste_suj_data[(28*i)+j]
          
    T0, p_values , H0  = mne.stats.permutation_t_test(fullTableTest_condition,npermut)
    plt.hist(H0)
    significant_freqs = p_values <= 0.05
    print(significant_freqs)
    
    readable_pValue_table = np.zeros(shape=(28,82)) 
    for i in range(28):#elec
        for j in range(82):#freq
            readable_pValue_table[i,j] = p_values[(82*i)+j]    
    return readable_pValue_table

readable_pValue_table_pendule =  get_pvalue_allElec_allFreq(liste_pendule,20000)   
np.savetxt("../csv_files/pvalueperm_allElec_allFreq_pendule_20000perm_fenetreModif.csv", readable_pValue_table_pendule,delimiter=",") 
np.savetxt('../csv_files/pvalueperm_allElec_allFreq_pendule_20000perm_fenetreModif.txt',readable_pValue_table_pendule,delimiter=",")

readable_pValue_table_main =  get_pvalue_allElec_allFreq(liste_main,20000)
np.savetxt("../csv_files/pvalueperm_allElec_allFreq_main_20000perm_fenetreModif.csv", readable_pValue_table_main,delimiter=",")       
np.savetxt('../csv_files/pvalueperm_allElec_allFreq_main_20000perm_fenetreModif.txt',readable_pValue_table_main,delimiter=",")

readable_pValue_table_mainIllusion =  get_pvalue_allElec_allFreq(liste_mainIllusion,20000) 
np.savetxt("../csv_files/pvalueperm_allElec_allFreq_mainIllusion_20000perm_fenetreModif.csv", readable_pValue_table_mainIllusion,delimiter=",")            
np.savetxt('../csv_files/pvalueperm_allElec_allFreq_mainIllusion_20000perm_fenetreModif.txt',readable_pValue_table_mainIllusion,delimiter=",")

#mtn on veut faire la meme chose mais avec une taille d'effet au lieu de p value
# d de cohen = moyGroup1 sur 23 sujets - moyGroup2(=0 pck test VS 0)
#divise par ecart type / rac(n)
def get_dcohen_allElec_allFreq(liste_condition):
    ndarray = np.zeros(shape=(28,82))
    for elec in range(28):
        mean = np.mean(liste_condition[elec],axis=0)
        print(mean)
        stdev = np.std(liste_condition[elec],axis=0)
        print(stdev)
        dcohen = mean/stdev
        ndarray[elec]=dcohen
            
    return ndarray

d_p = get_dcohen_allElec_allFreq(liste_pendule)
d_m = get_dcohen_allElec_allFreq(liste_main)
d_mi = get_dcohen_allElec_allFreq(liste_mainIllusion)
np.savetxt('../csv_files/dcohen_allElec_allFreq_pendule_fenetreModif.csv',d_p,delimiter=",")
np.savetxt('../csv_files/dcohen_allElec_allFreq_main_fenetreModif.csv',d_m,delimiter=",")
np.savetxt('../csv_files/dcohen_allElec_allFreq_mainIllusion_fenetreModif.csv',d_mi,delimiter=",")

import numpy as np 
import pandas as pd
mIll = pd.read_csv("C:/Users/claire.dussard/OneDrive - ICM/Bureau/old_papers/rdom_scriptsData/allElecFreq_VSZero/refait_25/dcohen_mIll.csv",header=None,delimiter=";")
main = pd.read_csv("C:/Users/claire.dussard/OneDrive - ICM/Bureau/old_papers/rdom_scriptsData/allElecFreq_VSZero/refait_25/dcohen_main.csv",header=None,delimiter=";")
pend = pd.read_csv("C:/Users/claire.dussard/OneDrive - ICM/Bureau/old_papers/rdom_scriptsData/allElecFreq_VSZero/refait_25/dcohen_pendule.csv",header=None,delimiter=";")



import imagesc
imagesc.plot(pend,cmap="Blues")
imagesc.plot(main,cmap="Blues")
imagesc.plot(mIll,cmap="Blues")

raw_signal.plot(block=True)

#creer le mask

p_pend = pd.read_csv("C:/Users/claire.dussard/OneDrive - ICM/Bureau/old_papers/rdom_scriptsData/allElecFreq_VSZero/refait_25/p_pend.csv",header=None)
p_main = pd.read_csv("C:/Users/claire.dussard/OneDrive - ICM/Bureau/old_papers/rdom_scriptsData/allElecFreq_VSZero/refait_25/p_main.csv",header=None)
p_mIll = pd.read_csv("C:/Users/claire.dussard/OneDrive - ICM/Bureau/old_papers/rdom_scriptsData/allElecFreq_VSZero/refait_25/p_mIll.csv",header=None)

p_pend = p_pend.to_numpy()
p_main = p_main.to_numpy()
p_mIll = p_mIll.to_numpy()

legends = pd.read_excel("C:/Users/claire.dussard/OneDrive - ICM/Bureau/old_papers/rdom_scriptsData/allElecFreq_VSZero/pvalue/pvalueperm_allElec_allFreq_main.xlsx")
elec_leg = legends["channel\\freq"]
imagesc.plot(p_pend)
imagesc.plot(p_main)
imagesc.plot(p_mIll)
raw_signal.plot(block=True)

# 1 avec un masque binaire
# pend = d_p
# main = d_m
# mIll = d_mi

pvalue = 0.05/3  
masked_p = np.ma.masked_where((p_pend > pvalue) , pend)
masked_m = np.ma.masked_where((p_main > pvalue) , main)
masked_mi = np.ma.masked_where((p_mIll > pvalue) , mIll)

imagesc.plot(-masked_p,cmap="Blues")
imagesc.plot(-masked_m,cmap="Blues")
imagesc.plot(-masked_mi,cmap="Blues")


#sans image sc
import matplotlib.pyplot as plt
gridspec_kw={'width_ratios': [1,1,1],
                           'height_ratios': [1],
                       'wspace': 0.05,#constrained_layout=True
                       'hspace': 0.05}
fig, axs = plt.subplots(1,3, sharey=True,sharex=True, figsize=(20, 7),gridspec_kw=gridspec_kw,constrained_layout=True)
vmin = 0.9
vmax = 2.1
img = axs[0].imshow(-masked_p, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="pendulum")
axs[0].text(0.12, 1.02, 'Virtual pendulum')

axs[1].imshow(-masked_m, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax)
axs[1].text(0.12, 1.02, 'Virtual hand')
axs[2].imshow(-masked_mi, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax)
axs[2].text(0.12, 1.02, 'Virtual hand with vibrations')
fig.colorbar(img, location = 'right')
elecs = elec_leg 
#plt.subplots_adjust(wspace=0.2, hspace=0.05)
freq_leg = np.arange(3,84,4)
freq_leg_str =[str(f) for f in freq_leg]
plt.xticks(np.linspace(0,1,21),freq_leg_str)
x8Hz = 0.061
x30Hz = 0.34
col = "black"
ls = "--"
lw = 0.7
for ax in axs.flat:
    ax.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
    ax.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for ax in axs.flat:
    for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
        ax.axhline(y=elecPos,color="dimgray",lw=0.25)
#plt.tight_layout(pad=0.04) 
raw_signal.plot(block=True)#specifier le x




#=======permutation cluster test=========
# listeSujetsPendule_C3 = []
# listeSujetsMain_C3 = []
# listeSujetsMainIllusion_C3 = []
# for sujet in listeNumSujetsFinale:#C3 8-30Hz
    
#     mat_p = scipy.io.loadmat('../MATLAB_DATA/'+sujet+'/'+sujet+'-penduletimePooled.mat')["data"]
#     listeSujetsPendule_C3.append(mat_p)
#     mat_m = scipy.io.loadmat('../MATLAB_DATA/'+sujet+'/'+sujet+'-maintimePooled.mat')["data"]
#     listeSujetsMain_C3.append(mat_m)
#     mat_mi = scipy.io.loadmat('../MATLAB_DATA/'+sujet+'/'+sujet+'-mainIllusiontimePooled.mat')["data"]
#     listeSujetsMainIllusion_C3.append(mat_mi)

# pval = 0.001  # arbitrary
# dfn = 3 - 1  # degrees of freedom numerator
# dfd = 23 - 3  # degrees of freedom denominator
# thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution

from mne.channels import find_ch_adjacency
#adjacency, ch_names = mne.channels.read_ch_adjacency("easycap32ch-avg")
#adjacency, ch_names = find_ch_adjacency(EpochDataMain[0].drop_channels(["TP9","TP10","FT9","FT10"]).info, ch_type='eeg')
#print(type(adjacency))  # it's a sparse matrix!

# fig, ax = plt.subplots(figsize=(5, 4))
# ax.imshow(adjacency.toarray(), cmap='gray', origin='lower',
#           interpolation='nearest')
# ax.set_xlabel('{} Magnetometers'.format(len(ch_names)))
# ax.set_ylabel('{} Magnetometers'.format(len(ch_names)))
# ax.set_title('Between-sensor adjacency')
# fig.tight_layout()
# raw_signal.plot(block=True)


# #add freqs to the adjacency matrix
# from mne.stats import combine_adjacency
# tfr_adjacency = combine_adjacency(
#     len(freqValues),adjacency)

# F_obs, clusters, p_values, _  = mne.stats.permutation_cluster_test(listeSujetsPendule_C3,
#                                                                                        threshold=thresh,n_permutations=10000)#liste d'elec x freq

# F_obs, clusters, p_values, _  = mne.stats.permutation_cluster_test(listeSujetsMain_C3,
#                                                                                        threshold=thresh,n_permutations=10000)#liste d'elec x freq
# F_obs, clusters, p_values, _  = mne.stats.permutation_cluster_test(listeSujetsMainIllusion_C3,
#                                                                                        threshold=thresh,n_permutations=10000)#liste d'elec x freq
# #display results

# fin du permutation cluster



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

