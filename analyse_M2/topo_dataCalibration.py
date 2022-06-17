# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:28:34 2022

@author: claire.dussard
"""
import os 
import seaborn as sns
import pathlib
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *
import numpy as np 

liste_essai = ["3-1", "3-1", "3-1", "3-1", "3-1", "3-1", "3-1", "3", "3-1", "3-1", "3", "3-1", "3-1", "3-1", "3",
                    "3-1", "3-1", "3-1", "3-1", "3-1", "3-1", "3-1", "3-1", "3-1", "3-1"]

#on commence au 3 (reste pbs noms)
liste_rawPath_calib = []
for num_sujet in allSujetsDispo_rest:
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
    raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + liste_essai[num_sujet]+".vhdr")#il faudrait recup les epochs et les grouper ?
    liste_rawPath_calib.append(raw_path_sample)
    
event_id={'Main':3} 
liste_epochsPreICA,liste_epochsSignal = pre_process_donnees(liste_rawPath_calib,1,0.1,90,[50,100],31,'Fz',event_id,10)#que 2 premiers sujets

listeICApreproc=[]
listeICA= []
for i in range(len(liste_epochsPreICA)):
    averageRefSignal_i,ICA_i = treat_indiv_data(liste_epochsPreICA[i],liste_epochsSignal[i],'Fz')
    listeICApreproc.append(averageRefSignal_i)
    listeICA.append(ICA_i)
 
#save_ICA_files(listeICA,liste_rawPath_calib,True)#a faire
#saveEpochsAfterICA_avantdropBad_windows(listeICApreproc,liste_rawPath_calib,True)

epochDataRest = load_data_postICA_preDropbad_effetFBseul(liste_rawPath_calib,"",True,True)

montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in listeICApreproc:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
        
EpochData = listeICApreproc
liste_power_sujets = []
freqs = np.arange(3, 85, 1)
n_cycles = freqs 
i = 0
for epochs_sujet in EpochData:
    print("========================\nsujet"+str(i))
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...")
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False)
    print("computing power...")
    liste_power_sujets.append(power_sujet)
    i += 1

#save_tfr_data(liste_power_sujets,liste_rawPath_calib,"",True)

#exemple de topomap
liste_power_sujets[0].plot_topomap(fmin=8,fmax=30,tmin=0,tmax=8,baseline=(-1,0),mode="logratio")
raw_signal.plot(block=True)

liste_power_sujets[1].plot_topomap(fmin=8,fmax=30,tmin=0,tmax=8,baseline=(-1,0),mode="logratio")

raw_signal.plot(block=True)
