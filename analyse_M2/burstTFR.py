# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:51:21 2022

@author: claire.dussard
"""

import os 
import pathlib
import mne
#necessite d'avoir execute handleData_subject.py, et load_savedData avant 
import numpy as np 
# importer les fonctions definies par moi 
from handleData_subject import createSujetsData
from functions.load_savedData import *
#import gc#garbage collector for memory leaks

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)


liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
       
jetes_pendule = [
    [],[],[],[5],[1,7,10],[],[],[3,5,8,10],[],[5,10],[],
    [5,6],[4],[6,9],[],[9],[3,8,9],[],[],[1,6],[6],[3,9],[6,8]
    ]

jetes_main = [
    [],[],[],[3],[2,4,5,6,7],[7],[],[6],[9,10],[8],[6],
    [1,6,8],[1,10],[9,10],[6,7,8,9,10],[3,6],[3,6,7],[4,10],[],[1,6],[],[9],[]
    ]

def load_indivEpochData(index_sujet,liste_rawPath):
    epoch_sujet = load_data_postICA_postdropBad_windows(liste_rawPath[index_sujet:index_sujet+1],"",True)[0]
    montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
    if epoch_sujet!=None:
        epoch_sujet.set_montage(montageEasyCap)
    return epoch_sujet
def compute_condition_power(i,liste_raw,freqs,n_cycles):
    print("je suis compute_condition_power")
    epochs_sujet = load_indivEpochData(i,liste_raw)
    print(epochs_sujet)
    print("downsampling...") #decim= 5 verifier si resultat pareil qu'avec down sampling
    epochs_sujet = epochs_sujet.resample(250., npad='auto') 
    print("computing power...")
    power_sujet = mne.time_frequency.tfr_morlet(epochs_sujet,freqs=freqs,n_cycles=n_cycles,return_itc=False,average=False,decim =5)#AVERAGE = FALSE : 1 par essai
    return power_sujet

# fmin = 8
# fmax = 40
# pasFreq = 1
# tmin = -5
# liste_TFR_pendule_allSujets = []
# liste_TFR_main_allSujets = []
# for suj in range(0,5):
#     num_sujet = suj
#     listeTFR_pendule,listeTFR_main = get_30essais_sujet_i(num_sujet,fmin,fmax,pasFreq)
#     liste_TFR_pendule_allSujets.append(listeTFR_pendule)
#     liste_TFR_main_allSujets.append(listeTFR_main)
    
liste_TFR_main_allSujets = load_tfr_data_windows(liste_rawPathMain[0:5],"allTrials",True)
liste_TFR_pendule_allSujets = load_tfr_data_windows(liste_rawPathPendule[0:5],"allTrials",True)
# save_tfr_data(liste_TFR_pendule_allSujets,liste_rawPathPendule,"allTrials",True)

def get_30essais_sujet_i(num_sujet,freqMin,freqMax,pasFreq):
    freqs = np.arange(freqMin, freqMax, pasFreq)
    n_cycles = freqs /2
    i = num_sujet
    power_sujet_pendule = compute_condition_power(i,liste_rawPathPendule,freqs,n_cycles)
    power_sujet_main = compute_condition_power(i,liste_rawPathMain,freqs,n_cycles)
    return power_sujet_pendule,power_sujet_main

liste_power_sujets = load_tfr_data_windows(liste_rawPath_rawRest,"",True)
#plot rest
fig, axs = plt.subplots(5)
i = 0
for power in liste_power_sujets:
    power.plot(picks="C3",vmin=0,vmax=60e-11,axes=axs[i],tmin=20,tmax=45,fmax=fmax,fmin=fmin)
    i += 1
raw_signal.plot(block=True) 



fmin = 8
fmax = 40
pasFreq = 1
tmin = -5

for suj in range(0,5):
    num_sujet = suj
    listeTFR_main = liste_TFR_main_allSujets[suj]
    listeTFR_pendule = liste_TFR_pendule_allSujets[suj]
    #listeTFR_pendule,listeTFR_main = get_30essais_sujet_i(num_sujet,fmin,fmax,pasFreq)
        
    fig, axs = plt.subplots(10,2)
    fig.suptitle('Vertically stacked subplots')
    delta = 0
    for i in range(10):
        print(i)
        if i+1 in jetes_pendule[num_sujet]:
            print("pass"+str(i))
            delta += 1
            pass
        else:
            listeTFR_pendule[i-delta].average().plot(picks="C3",vmin=0,vmax=60e-11,axes=axs[i,0],tmin=tmin,tmax=25,fmax=fmax,fmin=fmin)
    delta = 0
    for i in range(10):
        if i+1 in jetes_main[num_sujet]:
            print("pass"+str(i))
            delta += 1
            pass
        else:
            listeTFR_main[i-delta].average().plot(picks="C3",vmin=0,vmax=60e-11,axes=axs[i,1],tmin=tmin,tmax=25,fmax=fmax,fmin=fmin)
raw_signal.plot(block=True)   



listeTfrAv_exec = load_tfr_data_windows(liste_rawPathObsExec[13:],"exec",True)
my_cmap = discrete_cmap(13, 'Reds')
for tfr in listeTfrAv_exec:
    tfr.plot(picks="C3",vmin=0,vmax=4e-11,fmin=3,fmax=48,cmap=my_cmap)
raw_signal.plot(block=True)
data  = np.mean(listeTfrAv_exec[0].data[5][10:27],axis=0)
raw_signal.plot(block=True)

# listeTFR = get_30essais_sujet_i(22,8,48,1,jetes_pendule)
# fig, axs = plt.subplots(len(listeTFR))
# fig.suptitle('Vertically stacked subplots')

# for i in range(len(listeTFR)):
#     listeTFR[i].average().plot(picks="C3",vmin=0,vmax=4e-11,axes=axs[i])#vmin=-0.4,vmax=0.4,baseline=(-3,-1),mode="logratio")
# raw_signal.plot(block=True) 


 #effect of n_cycles
 
 def get_30essais_sujet_i_differentNcycles(num_sujet,freqMin,freqMax,pasFreq):
     freqs = np.arange(freqMin, freqMax, pasFreq)
     i = num_sujet
     power_freqs4 = compute_condition_power(i,liste_rawPathMain,freqs,freqs /4)
     power_sujet_freqs = compute_condition_power(i,liste_rawPathMain,freqs,freqs)
     power_sujet_20 = compute_condition_power(i,liste_rawPathMain,freqs,20)
     power_sujet_4 = compute_condition_power(i,liste_rawPathMain,freqs,4)
     return power_freqs4,power_sujet_freqs,power_sujet_20,power_sujet_4


power_freqs4,power_sujet_freqs,power_sujet_20,power_sujet_4 = get_30essais_sujet_i_differentNcycles(20,8,40,1)
fig, axs = plt.subplots(10,4)
for i in range(10):
    power_freqs4[i].average().plot(picks="C3",vmin=0,vmax=60e-11,axes=axs[i,0])
    power_sujet_freqs[i].average().plot(picks="C3",vmin=0,vmax=60e-11,axes=axs[i,1])
    power_sujet_20[i].average().plot(picks="C3",vmin=0,vmax=60e-11,axes=axs[i,2])
    power_sujet_4[i].average().plot(picks="C3",vmin=0,vmax=60e-11,axes=axs[i,3])
raw_signal.plot(block=True)



def plot_elec_cond_generalise(power_sujet,elec_name,cond,elec_position,freqs,fig,ax,ax_nb,scaleMin,scaleMax,tmin,tmax,fmin,fmax,color,label):
    delta = 0  
    av_power = power_sujet[i-delta]
    ch_names = av_power.info.ch_names
    if ch_names[elec_position]!=elec_name:
        print(av_power.info.ch_names[elec_position]+"IS NOT "+elec_name)
        elec_position = ch_names.index(elec_name)
    av_power_copy = av_power.copy()
    av_power_copy.crop(tmin=tmin,tmax=tmax)
    data = av_power_copy.data
    data_meanTps = np.mean(data,axis=0)
    data_meanTps = np.mean(data_meanTps,axis=2)
    data_elec = data_meanTps[elec_position][:]
    if label:
        if color is not None:
            ax[ax_nb,i].plot(freqs,data_elec, label=elec_name+" "+cond,color=color)
        else:
            ax[ax_nb,i].plot(freqs,data_elec, label=elec_name+" "+cond)
            
    else:
        ax[ax_nb,i].plot(freqs,data_elec,color=color) 
    plt.legend(loc="upper left")
    ax[ax_nb,i].axvline(x=8,color="black",linestyle="--")
    ax[ax_nb,i].axvline(x=30,color="black",linestyle="--")
    ax[ax_nb,i].set_xlim([fmin, fmax])
    ax[ax_nb,i].set_ylim([scaleMin, scaleMax])
        
#who has a peak ?
liste_TFR_main = load_tfr_data_windows(liste_rawPathMain,"",True)
liste_TFR_pendule = load_tfr_data_windows(liste_rawPathPendule,"",True)

liste_power_sujets = load_tfr_data_windows(liste_rawPath_rawRest,"",True)
for power in liste_power_sujets:
    power.plot(picks="C3",vmin=0,vmax = 10e-10)

fig, axs = plt.subplots(2,13)
# do they have rest peaks
i = 0
elec_name = "C3"
tmin = 5
tmax = 25
freqs = np.arange(4, 41, 1)
j = 0
sujet = 0
for power in liste_power_sujets:
    print(i)
    av_power = power
    ch_names = power.info.ch_names
    elec_position = ch_names.index(elec_name)
    av_power_copy = av_power.copy()
    av_power_copy.crop(tmin=tmin,tmax=tmax,fmin = 4,fmax = 40)
    data = av_power_copy.data
    data_elec = data[elec_position]
    data_meanTps = np.mean(data_elec,axis=1)
    axs[j,i].plot(freqs,data_meanTps, label="num sujet"+str(liste_rawPath_rawRest[sujet]).split("/")[0][0:8])
    axs[j,i].set_ylim([0, 10e-10])
    axs[j,i].legend()
    i += 1 
    sujet += 1
    if i ==12 and j!= 1:
        i = 0
        j += 1

raw_signal.plot(block=True)
