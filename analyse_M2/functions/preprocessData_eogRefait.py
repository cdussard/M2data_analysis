
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:20:54 2021

@author: claire.dussard
"""

import time
import matplotlib
import pathlib
import mne
import numpy as np
import PyQt5
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from mne.preprocessing import ICA

montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
def read_raw_data(listePath):
    print("reading data")
    listeRaw = []
    for path in listePath:
        print(path)
        raw = mne.io.read_raw_brainvision(path,preload=True,eog=('HEOG', 'VEOG'))#,misc=('EMG'))#AJOUT DU MISC EMG
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

    return listeFiltered

def montage_eeg(listeFilteredEEG):
    montageEasyCap = mne.channels.make_standard_montage('easycap-M1')#est ce que c'est le bon ?
    listeMontaged = []
    for recording in listeFilteredEEG:
        montaged=recording.set_montage(montageEasyCap)
        listeMontaged.append(montaged)
    return listeMontaged

def average_rereference(listeMontaged,initial_ref):
    listeAvReferenced = []#TO DO : ameliorer : verifier si les channels type FT9 sont pas >> amplitude, si oui exclure de l'av ou reconstruire puis average
    for recording in listeMontaged:
        #plot bad
        print(recording.info["bads"])
        signalInitialRef = mne.add_reference_channels(recording,initial_ref)
        averagedSignal = signalInitialRef.set_eeg_reference('average')
        listeAvReferenced.append(averagedSignal)
    return listeAvReferenced   


def ICA_preproc(liste_epochsICA,liste_epochsSignal,listeRawPath):
    listeSujetsRestants = listeRawPath.copy()
    liste_PostICA_epoch = []
    liste_ICA = []
    i = 1
    montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
    for i in range(len(liste_epochsICA)):#a remplacer
        if len(liste_epochsSignal[i])>1 and len(liste_epochsICA[i])>1:
            epochsICASujet = liste_epochsICA[i]
            epochsFiltered = liste_epochsSignal[i]
            epochsICASujet.set_montage(montageEasyCap) #eogSujet = liste_tous_epochs_EOG[i]
            print("iteration "+str(i)+"/"+str(len(liste_epochsSignal)))
            ica = ICA()#n_components=31)
            ica.fit(epochsICASujet) 
            print("Veuillez sélectionner les composantes à exclure sur le décours temporel :)")
            ica.plot_components(picks=['eeg']) #ne fonctionne que pour ica sur epoch unique
            print(ica.pca_explained_variance_)
            ica.plot_sources(epochsICASujet,block=True)#NE PAS METTRE STOP QUAND SUR EPOCHS  #eogSujet.plot(block=True,picks=['VEOG','HEOG'])
            reconst_raw_sujet = epochsFiltered.copy()
            ica.apply(reconst_raw_sujet)
            reconst_raw_sujet.plot(title='ICA- Reconstructed signal')
            epochsFiltered.plot(title = 'Filtered signal',block=True)      
            print("start appending")
            liste_PostICA_epoch.append(reconst_raw_sujet)
            liste_ICA.append(ica)
            print("done appending")
        else:
            listeSujetsRestants.remove(listeSujetsRestants[i])#pb les epochs droppes reviennent a l'ICA (pas ceux rejetes par artefact par contre)
    return liste_PostICA_epoch,liste_ICA,listeSujetsRestants
        
def epoching(event_id,listeFilteredICA,listeFilteredSignal,dureeEpoch,dureePreEpoch,reject):
    epochsCibles = []
    liste_epochsSignal = []
    liste_epochsPreICA = []
    for signal,signalICA in zip(listeFilteredSignal,listeFilteredICA):
         events = mne.events_from_annotations(signal)[0] #print(events)
         epochsCibles = mne.Epochs(signal,events,event_id,tmin=-dureePreEpoch,tmax = dureeEpoch,baseline=None, preload=True)#,reject=reject)#pas de picks avant ICA           
         epochsICA = mne.Epochs(signalICA,events,event_id,tmin=-dureePreEpoch,tmax = dureeEpoch,baseline=None, preload=True)#,reject=reject)
         #mark bad epochs for ICA 
         if len(epochsICA)>0:
             epochsICA.plot(block=True)#mark bad
         epochsICA.info["bads"] = epochsCibles.info["bads"]
         #instead of rejecting epoch, mark bad and drop if len(bad channels >6 ?) Fp1 & Fp2 merdent tjrs
         #epochsCibles.plot_drop_log()
         #print(epochsCibles)     
         liste_epochsSignal.append(epochsCibles)
         liste_epochsPreICA.append(epochsICA)
    return liste_epochsPreICA,liste_epochsSignal

def change_order_channels(channels,listeSignals):
    listeEpochs_bonOrdreChannels = []
    for signal in listeSignals:
        print(signal.ch_names)
        signal.copy().reorder_channels(channels)
    return bonOrdreSignals
    
def mark_bad_electrodes(listeSignals,listeElectrodesBad):
    if len(listeElectrodesBad)>0:
        for signal in listeSignals:
            signal.info['bads']=listeElectrodesBad
        print("marked bad")
        #les marquer bad
    else:
        #exclure basé sur l'amplitude
        print("t'as pas implemente la methode")
    
    return listeSignals
        
def pre_process_donnees(listeRawPath,low_freqICA,lowFreqSignal,high_freq,notch_freqs,n_ICA_components,initial_ref,event_id):
    channels = ['VEOG','HEOG','Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9','CP5','CP1','CP2','CP6','TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    listeRaw = read_raw_data(listeRawPath)
    #filtre a 1Hz le signal pour ICA mais a 0.1 Hz le signal a analyser
    listeFilteredSignal = filter_data(listeRaw,lowFreqSignal,high_freq,notch_freqs)
    listeFilteredICA = filter_data(listeRaw,low_freqICA,high_freq,notch_freqs)
    #mark bad
    listeFilteredSignal_bad = mark_bad_electrodes(listeFilteredSignal,['TP9','TP10'])#,'FT9','FT10'])
    listeFilteredICA_bad = mark_bad_electrodes(listeFilteredICA,['TP9','TP10'])#,'FT9','FT10'])
    #epoching
    dureeEpoch = 31#29.3#entre 28.6 et 31.1 24s en vrai
    dureePreEpoch = 5.0
    reject = dict(eeg=50e-5 # unit: V (EEG channels) & 100 on drop rien, a 10 on drop tout ?
              )
    liste_epochsPreICA,liste_epochsSignal = epoching(event_id,listeFilteredICA_bad,listeFilteredSignal_bad,dureeEpoch,dureePreEpoch,reject)
    #ICA
    liste_PostICA_epoch,liste_ICA,listeSujetsRestants = ICA_preproc(liste_epochsPreICA,liste_epochsSignal,listeRawPath)
    #ICA_preproc(listeEpochsPreICA_badElectrodes,listeEpochs_badElectrodes)
    #save ICA et save epochs post correction
    #re-reference
    listeAverageRef = average_rereference(liste_PostICA_epoch,initial_ref)
    #montage
    listeMontaged = montage_eeg(listeAverageRef)
    #change order electrodes
    #listeEpochs_bonOrdreChannels = change_order_channels(channels,listeAverageRef)

    return listeAverageRef,listeFilteredSignal,liste_ICA,listeSujetsRestants


def return_epochs(listeRawPath,lowFreqSignal,high_freq,notch_freqs,event_id,preload):
    print("len listerawPath "+str(len(listeRawPath)))
    listeRaw = read_raw_data(listeRawPath)
    print("len listeRaw"+str(len(listeRaw)))
    #filtre a 1Hz le signal pour ICA mais a 0.1 Hz le signal a analyser
    listeFilteredSignal = filter_data(listeRaw,lowFreqSignal,high_freq,notch_freqs)
    print(listeFilteredSignal)
    print("len listeFilteredSignal"+str(len(listeFilteredSignal)))
    #epoching
    dureeEpoch = 31#29.3#entre 28.6 et 31.1 24s en vrai
    dureePreEpoch = 5.0
    liste_epochsSignal = []
    for signal in listeFilteredSignal :
         events = mne.events_from_annotations(signal)[0]
         epochsCibles = mne.Epochs(signal,events,event_id,tmin=-dureePreEpoch,tmax = dureeEpoch,baseline=None, preload=preload)
         liste_epochsSignal.append(epochsCibles)
    print("len liste_epochsSignal"+str(len(liste_epochsSignal)))
    return liste_epochsSignal

def treat_indiv_data(epochsSujet,epochsPreICASujet,initial_ref):
    #==================================================================
    #plot epochs and exclude bad data segments / reconstruct it
    # add the code
    #==================================================================
    print('nb epochs Signal'+str(len(epochsSujet)))
    print('nb epochs ICA'+str(len(epochsPreICASujet)))
    if type(epochsSujet) is list:
        epochsSujet = mne.concatenate_epochs(epochsSujet)
        epochsPreICASujet = mne.concatenate_epochs(epochsPreICASujet)
    if len(epochsSujet)>0 and len(epochsPreICASujet)>0:
        # epochsSujet.info["bads"]=listeBad
        # epochsPreICASujet.info["bads"] = listeBad
        #epochsSujet.plot(block=True)#mark bad
        epochsPreICASujet.set_montage(montageEasyCap)
        #fit ICA
        ica = ICA()#n_components=31)
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
        #append??
    else:
        averageRefSignal = None
        ica = None
    return averageRefSignal,ica

def indiv_analysis(listeRawPath,lowFreqSignal,low_freqICA,high_freq,notch_freqs,event_id,initial_ref):
    montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
    listeRaw = read_raw_data(listeRawPath)
    #filtre a 1Hz le signal pour ICA mais a 0.1 Hz le signal a analyser
    listeFilteredSignal = filter_data(listeRaw,lowFreqSignal,high_freq,notch_freqs)
    listeFilteredICA = filter_data(listeRaw,low_freqICA,high_freq,notch_freqs)
    #mark bad
    listeBad = ["TP9","TP10","Fp1","Fp2"]#,"FT9","FT10"]#ft9 OrawPath_FBseulBLIGE pour sujets 10 a 15
    listeFilteredSignal_bad = mark_bad_electrodes(listeFilteredSignal,listeBad)#,'FT9','FT10'])
    listeFilteredICA_bad = mark_bad_electrodes(listeFilteredICA,listeBad)#,'FT9','FT10'])
    dureeEpoch = 31#29.3#entre 28.6 et 31.1 24s en vrai
    dureePreEpoch = 5.0
    reject = dict(
        eeg=50e-5 # unit: V (EEG channels) & 100 on drop rien, a 10 on drop tout ?
        )

    liste_epochsPreICA,liste_epochsSignal = epoching(event_id,listeFilteredICA_bad,listeFilteredSignal_bad,dureeEpoch,dureePreEpoch,reject)
    i = 0
    liste_ICA = []
    liste_epochs_Sujets = []
    for i in range(len(liste_epochsSignal)):#parcourir les sujets
        averageRefSignal,ICA = treat_indiv_data(liste_epochsSignal[i],liste_epochsPreICA[i],initial_ref) 
        #setMOntage
        if averageRefSignal is not None:
            averageRefSignal.set_montage(montageEasyCap) 
        liste_epochs_Sujets.append(averageRefSignal)
        liste_ICA.append(ICA)
    return liste_ICA,liste_epochs_Sujets

def all_conditions_analysis(allSujetsDispo,rawPath_main,rawPath_pendule,rawPath_mainIllusion,
                            event_id_main,event_id_pendule,event_id_mainIllusion,
                            lowFreqSignal,low_freqICA,high_freq,notch_freqs,initial_ref):
    if (len(rawPath_main)==len(rawPath_pendule)and len(rawPath_mainIllusion)==len(rawPath_main)):#all subjects have equal n#conditions
        nbSujets = len(rawPath_main)
        montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
        #========================Read raw data==================================================================
        listeRaw_main = read_raw_data(rawPath_main)
        listeRaw_pendule = read_raw_data(rawPath_pendule)
        listeRaw_mainIllusion = read_raw_data(rawPath_mainIllusion)
        #=======================Filter data=====================================================================
        listeFilteredSignal_main = filter_data(listeRaw_main,lowFreqSignal,high_freq,notch_freqs)
        listeFilteredICA_main = filter_data(listeRaw_main,low_freqICA,high_freq,notch_freqs)
        listeFilteredSignal_pendule = filter_data(listeRaw_pendule,lowFreqSignal,high_freq,notch_freqs)
        listeFilteredICA_pendule = filter_data(listeRaw_pendule,low_freqICA,high_freq,notch_freqs)
        listeFilteredSignal_mainIllusion = filter_data(listeRaw_mainIllusion,lowFreqSignal,high_freq,notch_freqs)
        listeFilteredICA_mainIllusion = filter_data(listeRaw_mainIllusion,low_freqICA,high_freq,notch_freqs)
        #mark bad
        #listeBad = ["TP9","TP10","Fp1","Fp2"]
        #======================Epoching data====================================================================
        dureeEpoch = 31#29.3#entre 28.6 et 31.1 24s en vrai
        dureePreEpoch = 5.0
        reject = dict(
            eeg=90e-5 # unit: V (EEG channels) & 100 on drop rien, a 10 on drop tout ?
            )
        liste_epochsPreICA_main,liste_epochsSignal_main = epoching(event_id_main,listeFilteredICA_main,listeFilteredSignal_main,dureeEpoch,dureePreEpoch,reject)
        liste_epochsPreICA_pendule,liste_epochsSignal_pendule = epoching(event_id_pendule,listeFilteredICA_pendule,listeFilteredSignal_pendule,dureeEpoch,dureePreEpoch,reject)
        liste_epochsPreICA_mainIllusion,liste_epochsSignal_mainIllusion = epoching(event_id_mainIllusion,listeFilteredICA_mainIllusion,listeFilteredSignal_mainIllusion,dureeEpoch,dureePreEpoch,reject)
        #=====================Treat individual subject data======================================================
        listeEpochs_main = []
        listeEpochs_pendule = []
        listeEpochs_mainIllusion = []
        listeICA_main = []
        listeICA_pendule = []
        listeICA_mainIllusion = []
        for i in range(nbSujets):
            print("\n===========Sujet S "+str(allSujetsDispo[i])+"========================\n")
            #================Compute ICA on runs=================================================================
            epochs_main,ICA_main = treat_indiv_data(liste_epochsSignal_main[i],liste_epochsPreICA_main[i],initial_ref) 
            epochs_pendule,ICA_pendule = treat_indiv_data(liste_epochsSignal_pendule[i],liste_epochsPreICA_pendule[i],initial_ref)
            epochs_mainIllusion,ICA_mainIllusion = treat_indiv_data(liste_epochsSignal_mainIllusion[i],liste_epochsPreICA_mainIllusion[i],initial_ref) 
            listeEpochs_main.append(epochs_main)
            listeICA_main.append(ICA_main)
            listeEpochs_pendule.append(epochs_pendule)
            listeICA_pendule.append(ICA_pendule)
            listeEpochs_mainIllusion.append(epochs_mainIllusion)
            listeICA_mainIllusion.append(ICA_mainIllusion)
           
    else:
        print("Subjects have unequal number of conditions, check your data")
    return listeEpochs_main,listeICA_main,listeEpochs_pendule,listeICA_pendule,listeEpochs_mainIllusion,listeICA_mainIllusion
    
def all_conditions_analysis_ICAload(allSujetsDispo,rawPath_main,rawPath_pendule,rawPath_mainIllusion,
                            event_id_main,event_id_pendule,event_id_mainIllusion,
                            lowFreqSignal,low_freqICA,high_freq,notch_freqs,initial_ref):
    if (len(rawPath_main)==len(rawPath_pendule)and len(rawPath_mainIllusion)==len(rawPath_main)):#all subjects have equal n#conditions
        nbSujets = len(rawPath_main)
        montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
        epochsMain = return_epochs(rawPath_main,lowFreqSignal,high_freq,notch_freqs,event_id_main,preload=True)
        epochsPendule = return_epochs(rawPath_pendule,lowFreqSignal,high_freq,notch_freqs,event_id_pendule,preload=True)
        epochsMainIllusion = return_epochs(rawPath_mainIllusion,lowFreqSignal,high_freq,notch_freqs,event_id_mainIllusion,preload=True)
        #=====================Treat individual subject data======================================================
        listeEpochs_main = []
        listeEpochs_pendule = []
        listeEpochs_mainIllusion = []
        #load ICA data
        liste_ICA_main = load_ICA_files(rawPath_main)
        liste_ICA_pendule = load_ICA_files(rawPath_pendule)
        liste_ICA_mainIllusion = load_ICA_files(rawPath_mainIllusion)
        for i in range(nbSujets):
            print(epochsMain)
            print(epochsMain[i])
            print("\n===========Sujet S "+str(allSujetsDispo[i])+"========================\n")
            #================Compute ICA on runs=================================================================
            epochsMain_sujet_ica = liste_ICA_main[i].apply(epochsMain[i])
            epochsPendule_sujet_ica = liste_ICA_pendule[i].apply(epochsPendule[i])
            epochsMainIllusion_sujet_ica = liste_ICA_mainIllusion[i].apply(epochsMainIllusion[i])
            listeEpochs_main.append(epochsMain_sujet_ica)
            listeEpochs_pendule.append(epochsPendule_sujet_ica)
            listeEpochs_mainIllusion.append(epochsMainIllusion_sujet_ica)
           
    else:
        print("Subjects have unequal number of conditions, check your data")
    return listeEpochs_main,listeEpochs_pendule,listeEpochs_mainIllusion
    


def all_conditions_analysis_FBseul(allSujetsDispo,rawPath_FBseul,
                            event_id_main,event_id_pendule,event_id_mainIllusion,
                            lowFreqSignal,low_freqICA,high_freq,notch_freqs,initial_ref):

    nbSujets = len(rawPath_FBseul)
    montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
    #========================Read raw data==================================================================
    listeRaw_FBseul = read_raw_data(rawPath_FBseul)
    #=======================Filter data=====================================================================
    listeFilteredSignal_FBseul = filter_data(listeRaw_FBseul,lowFreqSignal,high_freq,notch_freqs)
    listeFilteredICA_FBseul = filter_data(listeRaw_FBseul,low_freqICA,high_freq,notch_freqs)
    #======================Epoching data====================================================================
    dureeEpoch = 30.0#29.3#entre 28.6 et 31.1 24s en vrai
    dureePreEpoch = 5.0
    reject = dict(
        eeg=90e-5 # unit: V (EEG channels) & 100 on drop rien, a 10 on drop tout ?
        )
    ListeEpochPreICA_main,ListeEpochSignal_main = epoching(event_id_main,listeFilteredICA_FBseul,listeFilteredSignal_FBseul,dureeEpoch,dureePreEpoch,reject)
    ListeEpochPreICA_pendule,ListeEpochSignal_pendule = epoching(event_id_pendule,listeFilteredICA_FBseul,listeFilteredSignal_FBseul,dureeEpoch,dureePreEpoch,reject)
    ListeEpochPreICA_mainIllusion,ListeEpochSignal_mainIllusion = epoching(event_id_mainIllusion,listeFilteredICA_FBseul,listeFilteredSignal_FBseul,dureeEpoch,dureePreEpoch,reject)
    #=====================Treat individual subject data======================================================
    listeEpochs_FBseul= []
    listePreICA_FBseul = []
    for i in range(nbSujets):
        listeEpochs_FBseul.append([ListeEpochSignal_main[i],ListeEpochSignal_pendule[i],ListeEpochSignal_mainIllusion[i]])
        listePreICA_FBseul.append([ListeEpochPreICA_main[i],ListeEpochPreICA_pendule[i],ListeEpochPreICA_mainIllusion[i]])
    listeEpochs_main = []
    listeEpochs_pendule = []
    listeEpochs_mainIllusion = []
    listeICA = []
    for i in range(nbSujets):
        print("\n===========Sujet S "+str(allSujetsDispo[i])+"========================\n")
        #================Compute ICA on runs=================================================================
        all_epochs,all_ICA = treat_indiv_data(listeEpochs_FBseul[i],listePreICA_FBseul[i],initial_ref) 
        listeEpochs_main.append(all_epochs[0])
        listeICA.append(all_ICA)
        listeEpochs_pendule.append(all_epochs[1])
        listeEpochs_mainIllusion.append(all_epochs[2])
       
    return listeEpochs_main,listeICA,listeEpochs_pendule,listeEpochs_mainIllusion
    
def all_conditions_analysis_NFBRest(allSujetsDispo,rawPath_main,rawPath_mainIllusion,
                            event_id_main,event_id_mainIllusion,
                            lowFreqSignal,low_freqICA,high_freq,notch_freqs,initial_ref):
    if len(rawPath_mainIllusion)==len(rawPath_main):#all subjects have equal n#conditions
        nbSujets = len(rawPath_main)
        montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
        #========================Read raw data==================================================================
        listeRaw_main = read_raw_data(rawPath_main)
        listeRaw_mainIllusion = read_raw_data(rawPath_mainIllusion)
        #=======================Filter data=====================================================================
        listeFilteredSignal_main = filter_data(listeRaw_main,lowFreqSignal,high_freq,notch_freqs)
        listeFilteredICA_main = filter_data(listeRaw_main,low_freqICA,high_freq,notch_freqs)
        listeFilteredSignal_mainIllusion = filter_data(listeRaw_mainIllusion,lowFreqSignal,high_freq,notch_freqs)
        listeFilteredICA_mainIllusion = filter_data(listeRaw_mainIllusion,low_freqICA,high_freq,notch_freqs)
        #mark bad
        #listeBad = ["TP9","TP10","Fp1","Fp2"]
        #======================Epoching data====================================================================
        dureeEpoch = 31#29.3#entre 28.6 et 31.1 24s en vrai
        dureePreEpoch = 5.0
        reject = dict(
            eeg=90e-5 # unit: V (EEG channels) & 100 on drop rien, a 10 on drop tout ?
            )
        liste_epochsPreICA_main,liste_epochsSignal_main = epoching(event_id_main,listeFilteredICA_main,listeFilteredSignal_main,dureeEpoch,dureePreEpoch,reject)
        liste_epochsPreICA_mainIllusion,liste_epochsSignal_mainIllusion = epoching(event_id_mainIllusion,listeFilteredICA_mainIllusion,listeFilteredSignal_mainIllusion,dureeEpoch,dureePreEpoch,reject)
        #=====================Treat individual subject data======================================================
        listeEpochs_main = []
        listeEpochs_mainIllusion = []
        listeICA_main = []
        listeICA_mainIllusion = []
        for i in range(nbSujets):
            print("\n===========Sujet S "+str(allSujetsDispo[i])+"========================\n")
            #================Compute ICA on runs=================================================================
            epochs_main,ICA_main = treat_indiv_data(liste_epochsSignal_main[i],liste_epochsPreICA_main[i],initial_ref) 
            epochs_mainIllusion,ICA_mainIllusion = treat_indiv_data(liste_epochsSignal_mainIllusion[i],liste_epochsPreICA_mainIllusion[i],initial_ref) 
            listeEpochs_main.append(epochs_main)
            listeICA_main.append(ICA_main)
            listeEpochs_mainIllusion.append(epochs_mainIllusion)
            listeICA_mainIllusion.append(ICA_mainIllusion)
           
    else:
        print("Subjects have unequal number of conditions, check your data")
    return listeEpochs_main,listeICA_main,listeEpochs_mainIllusion,listeICA_mainIllusion


def baselineRest_analysis(allSujetsDispo,rawPath_BL1,rawPath_BL2,
                            event_id_baseline,lowFreqSignal,low_freqICA,high_freq,notch_freqs,initial_ref):
    if len(rawPath_BL1)==len(rawPath_BL2):#all subjects have equal n#conditions
        nbSujets = len(rawPath_BL2)
        montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
        #========================Read raw data==================================================================
        rawPathBl = rawPath_BL1 + rawPath_BL2
        listeRaw_baseline1 = read_raw_data(rawPath_BL1)
        listeRaw_baseline2 = read_raw_data(rawPath_BL2)
        #=======================Filter data=====================================================================
        listeFilteredSignal_bl1 = filter_data(listeRaw_baseline1,lowFreqSignal,high_freq,notch_freqs)
        listeFilteredICA_bl1 = filter_data(listeRaw_baseline1,low_freqICA,high_freq,notch_freqs)
        listeFilteredSignal_bl2 = filter_data(listeRaw_baseline2,lowFreqSignal,high_freq,notch_freqs)
        listeFilteredICA_bl2 = filter_data(listeRaw_baseline2,low_freqICA,high_freq,notch_freqs)
        #======================Epoching data====================================================================
        dureeEpoch = 110#29.3#entre 28.6 et 31.1 24s en vrai
        dureePreEpoch = 3.0
        reject = dict(
            eeg=90e-5 # unit: V (EEG channels) & 100 on drop rien, a 10 on drop tout ?
            )
        liste_epochsPreICA_bl1,liste_epochsSignal_bl1 = epoching(event_id_baseline,listeFilteredICA_bl1,listeFilteredSignal_bl1,dureeEpoch,dureePreEpoch,None)
        liste_epochsPreICA_bl2,liste_epochsSignal_bl2 = epoching(event_id_baseline,listeFilteredICA_bl2,listeFilteredSignal_bl2,dureeEpoch,dureePreEpoch,None)
        #=====================Treat individual subject data======================================================
        listeEpochs_baseline = []
        listeICA_baseline = []
        for i in range(nbSujets):
            print("\n===========Sujet S "+str(allSujetsDispo[i])+"========================\n")
            #================Compute ICA on runs=================================================================
            epochs_bl1,ICA_bl1 = treat_indiv_data(liste_epochsSignal_bl1[i],liste_epochsPreICA_bl1[i],initial_ref) 
            epochs_bl2,ICA_bl2 = treat_indiv_data(liste_epochsSignal_bl2[i],liste_epochsPreICA_bl2[i],initial_ref) 
            listeEpochs_baseline.append([epochs_bl1,epochs_bl2])
            listeICA_baseline.append([ICA_bl1,ICA_bl2])
           
    else:
        print("Subjects have unequal number of conditions, check your data")
    return listeEpochs_baseline,listeICA_baseline,#liste_epochsPreICA_bl1,liste_epochsSignal_bl1,liste_epochsPreICA_bl2,liste_epochsSignal_bl2 #