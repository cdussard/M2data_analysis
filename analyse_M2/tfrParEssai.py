#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 18:25:07 2022

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
       
def load_indivEpochData(index_sujet,liste_rawPath):
    epoch_sujet = load_data_postICA_postdropBad_windows(liste_rawPath[index_sujet:index_sujet+1],"",True)[0]
    montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
    if epoch_sujet!=None:
        epoch_sujet.set_montage(montageEasyCap)
    return epoch_sujet
  
#==========acceder aux essais jetes par sujet============
jetes_main = [
    [],[],[],[3],[2,4,5,6,7],[7],[],[6],[9,10],[8],[6],
    [1,6,8],[1,10],[9,10],[6,7,8,9,10],[3,6],[3,6,7],[4,10],[],[1,6],[],[9],[]
    ]

jetes_pendule = [
    [],[],[],[5],[1,7,10],[],[],[3,5,8,10],[],[5,10],[],
    [5,6],[4],[6,9],[],[9],[3,8,9],[],[],[1,6],[6],[3,9],[6,8]
    ]

jetes_mainIllusion = [
    [6],[1,3,6],[1,2],[],[5,6,8,9,10],[],[],[1,6,7,8],[6,7,8,9,10],[4,10],[1],
    [],[1,8,10],[10],[6,9],[9],[4,8,9],[4,8],[],[1,6],[],[1],[]
    ]

def compute_condition_power(i,liste_raw,essaisJetes,freqs,n_cycles):
    print("je suis compute_condition_power")
    epochs_sujet = load_indivEpochData(i,liste_raw)
    print(epochs_sujet)
    print("downsampling...") #decim= 5 verifier si resultat pareil qu'avec down sampling
    epochs_sujet = epochs_sujet.resample(250., npad='auto') 
    print("computing power...")
    power_sujet = mne.time_frequency.tfr_morlet(epochs_sujet,freqs=freqs,n_cycles=n_cycles,return_itc=False,average=False)#AVERAGE = FALSE : 1 par essai
    # del epochs_sujet
    # gc.collect()
    return power_sujet

def plot_condition_power(power_sujet,num_ax,essaisJetes,baseline,mode,freqMin,freqMax,tmin,tmax,axs,vmin,vmax,my_cmap):
    print("je suis plot_condition_power")
    delta = 0  
    print("essais jetes plot")
    print(essaisJetes)
    for i in range(10):
        if i+1 not in essaisJetes:#gerer les epochs jetes dans l'affichage
            power_sujet[i-delta].average().plot_topomap(baseline=baseline,mode=mode,fmin=freqMin,fmax=freqMax,tmin=2,tmax=25,axes=axs[num_ax,i],vmin=vmin,vmax=vmax,cmap=my_cmap)
        else:
            print("essai jete num"+str(i+1))
            delta += 1
            
    return True

def get_30essais_sujet_i(num_sujet,freqMin,freqMax,pasFreq,essaisJetes_main_sujet,essaisJetes_mainIllusion_sujet,essaisJetes_pendule_sujet):
    freqs = np.arange(3, 85, pasFreq)
    n_cycles = freqs 
    i = num_sujet
    my_cmap = discrete_cmap(13, 'RdBu_r')
    fig,axs = plt.subplots(3,10)
    vmin = -0.75
    vmax = 0.75
    tmin = 2
    tmax = 25
    dureePreBaseline = 3 #3
    dureePreBaseline = - dureePreBaseline
    dureeBaseline = 2.0 #2.0
    valeurPostBaseline = dureePreBaseline + dureeBaseline
    baseline = (dureePreBaseline, valeurPostBaseline)
    mode = 'zscore'    
    freqs = np.arange(3, 85, 1)
    n_cycles = freqs 
    #pendule
    power_sujet_pendule = compute_condition_power(i,liste_rawPathPendule,essaisJetes_pendule_sujet,freqs,n_cycles)
    plot_condition_power(power_sujet_pendule,0,essaisJetes_pendule_sujet,baseline,mode,freqMin,freqMax,tmin,tmax,axs,vmin,vmax,my_cmap)
    
    #main
    power_sujet_main = compute_condition_power(i,liste_rawPathMain,essaisJetes_main_sujet,freqs,n_cycles)
    plot_condition_power(power_sujet_main,1,essaisJetes_main_sujet,baseline,mode,freqMin,freqMax,tmin,tmax,axs,vmin,vmax,my_cmap)

    #mainIllusion
    power_sujet_mainIllusion = compute_condition_power(i,liste_rawPathMainIllusion,essaisJetes_mainIllusion_sujet,freqs,n_cycles)
    plot_condition_power(power_sujet_mainIllusion,2,essaisJetes_mainIllusion_sujet,baseline,mode,freqMin,freqMax,tmin,tmax,axs,vmin,vmax,my_cmap)
    if num_sujet>2:
        num_sujet_reel = num_sujet + 2
    elif num_sujet == 0   :
           num_sujet_reel = 0
    elif num_sujet <3:
        num_sujet_reel = num_sujet + 1
 
   
    fig.suptitle('Sujet n°'+str(num_sujet_reel), fontsize=16)
    return fig

# get_30essais_sujet_i(num_sujet=22,freqMin=3,freqMax=85,pasFreq=1,
#                      sujets_epochs_jetes_main=jetes_main,sujets_epochs_jetes_mainIllusion=jetes_mainIllusion,sujets_epochs_jetes_pendule=jetes_pendule)
# raw_signal.plot(block=True)

figs_stock = []
for i in range(10):#5 par 5 pour ne pas saturer la RAM
    print(i)
    suj_jetes_main = jetes_main[i]
    suj_jetes_mainIllusion = jetes_mainIllusion[i]
    suj_jetes_pendule = jetes_pendule[i]
    print("jete main")
    print(suj_jetes_main)
    print("jete mainIllusion")
    print(suj_jetes_mainIllusion)
    print("jete pendule")
    print(suj_jetes_pendule)
    figs_stock.append(get_30essais_sujet_i(num_sujet=i,freqMin=8,freqMax=30,pasFreq=1,
                     essaisJetes_main_sujet=suj_jetes_main,essaisJetes_mainIllusion_sujet=suj_jetes_mainIllusion,essaisJetes_pendule_sujet=suj_jetes_pendule))
num_sujet = 19
sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
sample_data_dir = pathlib.Path(sample_data_loc)
date_nom_fichier = dates[num_sujet][-4:]+"-"+dates[num_sujet][3:5]+"-"+dates[num_sujet][0:2]
raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + "_7-2-b.vhdr")

raw_signal = mne.io.read_raw_brainvision(raw_path_sample,preload=False,eog=('HEOG', 'VEOG'))

raw_signal.plot(block=True)

#diagnose memory leak
# import tracemalloc
# tracemalloc.start(10)
# get_30essais_sujet_i(num_sujet=7,freqMin=8,freqMax=30,pasFreq=1,
#                       essaisJetes_main_sujet=jetes_main[7],essaisJetes_mainIllusion_sujet=jetes_mainIllusion[7],essaisJetes_pendule_sujet=jetes_pendule[7])
# snapshot = tracemalloc.take_snapshot()  
# top_stats = snapshot.statistics('lineno')

# print("[ Top 10 ]")
# for stat in top_stats[:10]:
#     print(stat) 
# top_n(25, snapshot, trace_type='filename')
    
# def collect_stats(self):        
#     self.snapshots.append(tracemalloc.take_snapshot())        
#     if len(self.snapshots)>1: 
#         stats = self.snapshots[-1].filter_traces(filters).compare_to(self.snapshots[-2], 'filename')    

# for stat in stats[:10]:                
#     print("{} new KiB {} total KiB {} new {} total memory blocks: ".format(stat.size_diff/1024, stat.size / 1024, stat.count_diff ,stat.count))                
# for line in stat.traceback.format():                    
#     print(line)

# def plot_all_conditions_power(freqMin,freqMax,tmin,tmax,vmin,vmax,my_cmap,essaisJetes_main_sujet,essaisJetes_mainIllusion_sujet,essaisJetes_pendule_sujet,liste_rawPathPendule,liste_rawPathMain,liste_rawPathMainIllusion):
#     fig,axs = plt.subplots(3,10)
#     dureePreBaseline = 3 #3
#     dureePreBaseline = - dureePreBaseline
#     dureeBaseline = 2.0 #2.0
#     valeurPostBaseline = dureePreBaseline + dureeBaseline
#     baseline = (dureePreBaseline, valeurPostBaseline)
#     mode = 'zscore'    
#     freqs = np.arange(3, 85, 1)
#     n_cycles = freqs 
#     power_sujet_pendule = compute_condition_power(i,liste_rawPathPendule,essaisJetes_pendule_sujet,freqs,n_cycles)
#     plot_condition_power(power_sujet_pendule,0,essaisJetes_pendule_sujet,baseline,mode,freqMin,freqMax,tmin,tmax,axs,vmin,vmax,my_cmap)
    
#     #main
#     power_sujet_main = compute_condition_power(i,liste_rawPathMain,essaisJetes_main_sujet,freqs,n_cycles)
#     plot_condition_power(power_sujet_main,1,essaisJetes_main_sujet,baseline,mode,freqMin,freqMax,tmin,tmax,axs,vmin,vmax,my_cmap)

#     #mainIllusion
#     power_sujet_mainIllusion = compute_condition_power(i,liste_rawPathMainIllusion,essaisJetes_mainIllusion_sujet,freqs,n_cycles)
#     plot_condition_power(power_sujet_mainIllusion,2,essaisJetes_mainIllusion_sujet,baseline,mode,freqMin,freqMax,tmin,tmax,axs,vmin,vmax,my_cmap)
#     return fig,axs
#     #avoid memory leakage
        

    

