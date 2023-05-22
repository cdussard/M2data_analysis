# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:19:03 2022

@author: claire.dussard
"""
import os 
import seaborn as sns
import pathlib
import numpy as np
import mne

#se placer dans le bon directory

#definir les evenements avec lesquels on epoch
eventObs = {'Observation mvt':16}

#exemple pour lire un fichier et recuperer les evenements
raw_signal = mne.io.read_raw_gdf("put_rawPathToOneFile.gdf",preload=True)#eog=('HEOG', 'VEOG'))
 #l'une des 2 fonctions selon comment les events sont codes
#events = mne.find_events(raw_signal)
#events = mne.events_from_annotations(raw_signal)

#dans liste_rawPath, c'est une liste des path vers les fichiers .gdf pour toi du coup
liste_rawPath_aConstruire = []

#======fonctions pour traitement des donnees==
def read_raw_data(listePath):
    print("reading data")
    listeRaw = []
    for path in listePath:
        print(path)
        raw = mne.io.read_raw_gdf(path,preload=True,eog=('HEOG', 'VEOG'))
        listeRaw.append(raw)
    return listeRaw

def filter_data(listeRaw,low_freq,high_freq,notch_freqs):
    print("filtering data")
    listeFiltered = []
    for recording in listeRaw:
        recording_eeg = recording.copy().drop_channels(['ECG' ,'ACC_X','ACC_Z','ACC_Y'])#,'HEOG','VEOG'])'EMG'
        low_pass_eeg = recording_eeg.copy().filter(low_freq,None, method='iir', iir_params=dict(ftype='butter', order=4))
        high_pass_eeg = low_pass_eeg.copy().filter(None,high_freq, method='iir', iir_params=dict(ftype='butter', order=4))
        filtered = high_pass_eeg.notch_filter(freqs=notch_freqs, filter_length='auto',phase='zero')
        filtered.set_channel_types({'EMG': 'emg'}) 
        listeFiltered.append(filtered)
        
def epoching(event_id,listeFilteredICA,listeFilteredSignal,dureeEpoch,dureePreEpoch):
    epochsCibles = []
    liste_epochsSignal = []
    liste_epochsPreICA = []
    i = 0
    for signal,signalICA in zip(listeFilteredSignal,listeFilteredICA):
         print(i)
         events = mne.events_from_annotations(signal)[0] #print(events)
         epochsCibles = mne.Epochs(signal,events,event_id,tmin=-dureePreEpoch,tmax = dureeEpoch,baseline=None, preload=True)#,reject=reject)#pas de picks avant ICA           
         epochsICA = mne.Epochs(signalICA,events,event_id,tmin=-dureePreEpoch,tmax = dureeEpoch,baseline=None, preload=True)#,reject=reject)
         #mark bad epochs for ICA 
         if len(epochsICA)>0:
             epochsICA.plot(block=True)#mark bad
         epochsICA.info["bads"] = epochsCibles.info["bads"]
         liste_epochsSignal.append(epochsCibles)
         liste_epochsPreICA.append(epochsICA)
         i += 1
    return liste_epochsPreICA,liste_epochsSignal

def pre_process_donnees_batch(listeRawPath,low_freqICA,lowFreqSignal,high_freq,notch_freqs,n_ICA_components,initial_ref,liste_event_id,epochDuration):
    liste_epochsPreICA = []
    liste_epochsSignal = []
    channels = ['VEOG','HEOG','Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9','CP5','CP1','CP2','CP6','TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    listeRaw = read_raw_data(listeRawPath)
    #filtre a 1Hz le signal pour ICA mais a 0.1 Hz le signal a analyser
    listeFilteredSignal = filter_data(listeRaw,lowFreqSignal,high_freq,notch_freqs)
    listeFilteredICA = filter_data(listeRaw,low_freqICA,high_freq,notch_freqs)
    #epoching
    dureeEpoch = epochDuration#29.3#entre 28.6 et 31.1 24s en vrai
    dureePreEpoch = 5.0
    for event_id in liste_event_id:
        liste_epochsPreICA_temp,liste_epochsSignal_temp = epoching(event_id,listeFilteredICA,listeFilteredSignal,dureeEpoch,dureePreEpoch)
        liste_epochsPreICA.append(liste_epochsPreICA_temp)
        liste_epochsSignal.append(liste_epochsSignal_temp)
    return liste_epochsPreICA,liste_epochsSignal



liste_epochsPreICA,liste_epochsSignal = pre_process_donnees_batch(liste_rawPath_aConstruire,1,0.1,90,[50,100],31,'Fz',[eventObs],30)#que 2 premiers sujets
#tu recupere 2 listes d'epochs , une sur laquelle on fera tourner l'ICA, l'autre sur laquelle on appliquera l'ICA

#pour faire tourner l'ICA  

#======fonction ICA===========
def treat_indiv_data(epochsSujet,epochsPreICASujet,initial_ref):

    print('nb epochs Signal'+str(len(epochsSujet)))
    print('nb epochs ICA'+str(len(epochsPreICASujet)))
    if type(epochsSujet) is list:
        epochsSujet = mne.concatenate_epochs(epochsSujet)
        epochsPreICASujet = mne.concatenate_epochs(epochsPreICASujet)
    if len(epochsSujet)>0 and len(epochsPreICASujet)>0:
        epochsPreICASujet.set_montage(montageEasyCap)
        #fit ICA
        ica = mne.preprocessing.ICA()#n_components=31)
        ica.fit(epochsPreICASujet) 
        print("Veuillez sélectionner les composantes à exclure sur le décours temporel :)")
        ica.plot_components(picks=['eeg']) #ne fonctionne que pour ica sur epoch unique
        print(ica.pca_explained_variance_)
        ica.plot_sources(epochsPreICASujet,block=True)
        reconst_raw_sujet = epochsSujet.copy()
        ica.apply(epochsSujet)
        epochsSujet.plot(title='ICA- Reconstructed signal')
        reconst_raw_sujet.plot(title = 'Filtered signal',block=True)   
        #average ref
        signalInitialRef = mne.add_reference_channels(epochsSujet,initial_ref)
        averageRefSignal = signalInitialRef.set_eeg_reference('average')
    else:
        averageRefSignal = None
        ica = None
    return averageRefSignal,ica
#=================================

  
listeICApreproc=[]
listeICA= []
for i in range(len(liste_preICA)):
    averageRefSignal_i,ICA_i = treat_indiv_data(liste_epochsPreICA[i],liste_epochsSignal[i],'Fz')
    listeICApreproc.append(averageRefSignal_i)
    listeICA.append(ICA_i)
    
#tu recupere la liste des epochs post ICA (listeICApreproc) et les objets ICA (listeICA)

#pour mettre le montage spatial sur les epochs (mettre ton modele de casque)
#il faut le faire avant le calcul des temps-frequence
montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in listeICApreproc:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
        
EpochData= listeICApreproc
liste_power_sujets = []

freqs = np.arange(3, 85, 1)#toutes les freq entre 3 et 85Hz avec pas de 1
n_cycles = freqs 

i = 0
for epochs_sujet in EpochData:
    print("========================\nsujet"+str(i))
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') #downsample a 250Hz des donnees
    print("downsampling...")
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles)#calcul de la puissance
    print("computing power...")
    liste_power_sujets.append(power_sujet)
    i += 1      
              
#appliquer une baseline
for tfr in liste_power_sujets:
    tfr.apply_baseline(baseline=(-4,-2), mode='logratio', verbose=None)   
    
gd_average_obs = mne.grand_average(liste_power_sujets)

#exemple de visualisation
gd_average_obs.plot(picks="C3",fmax=80,vmin=-0.6,vmax=0.6)

raw_signal.plot(block=True)
