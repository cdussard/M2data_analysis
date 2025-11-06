# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:04:35 2022

@author: claire.dussard
"""
from pathlib import Path
from functions.preprocessData_eogRefait import *

channelsSansFz = ['Fp1', 'Fp2', 'F7', 'F3','F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10','T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5','CP1','CP2','CP6','TP10','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2','HEOG','VEOG']

path = Path('sub-S022/ses-20210512/eeg/BETAPARK_2021-05-12_9-2.vhdr')
raw_signal = read_raw_data([path])[0]
#raw_signal = read_raw_data(liste_rawPathMain[4:5])[0]
bonOrdreEEG=raw_signal.copy().reorder_channels(channelsSansFz)
bonOrdreEEG.plot(block=True,n_channels=33)#press B to show channels

event_id_main={'Essai_main':3}  
liste_epochsPreICA,liste_epochsSignal = pre_process_donnees_batch([path],1,0.1,90,[50,100],'Fz',[event_id_main],30)#liste_epochsSignal[0][0].reorder_channels(channelsSansFz).plot(block=True)
#liste_epochsPreICA,liste_epochsSignal = pre_process_donnees_batch(liste_rawPathMain[4:5],1,0.1,90,[50,100],'Fz',[event_id_main],30)#liste_epochsSignal[0][0].reorder_channels(channelsSansFz).plot(block=True)


liste_epochsSignal[0][0].info.bads=[]
liste_epochsSignal[0][0].pick_types(eeg=True)

ICA_preprocessed = treat_indiv_data(liste_epochsSignal[0][0],liste_epochsPreICA[0][0],'Fz')
    
downsampled = ICA_preprocessed[0].copy().resample(250., npad='auto') 
montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
downsampled.set_montage(montageEasyCap)
downsampled.plot(block=True)


liste_power_sujets = []
freqs = np.arange(3, 85, 1)
n_cycles = freqs 
i = 0
power_sujet = mne.time_frequency.tfr_morlet(downsampled,freqs=freqs,n_cycles=n_cycles,return_itc=False)

my_cmap = discrete_cmap(13, 'Reds')
power_sujet.plot_topo(vmin=0,vmax=1e-9,cmap=my_cmap,fmax=40)
power_sujet.plot_topo(vmin=-0.4,vmax=0.4,cmap=discrete_cmap(13, 'RdBu_r'),fmax=40,baseline=(-3,-1),mode="logratio")
downsampled.plot(block=True)

#same with NFB data
my_cmap = discrete_cmap(13, 'Reds')
av_power_main_noBL.plot_topo(vmin=0,vmax=1e-9,cmap=my_cmap,fmax=40)
av_power_main_noBL.plot_topo(vmin=-0.4,vmax=0.4,cmap=discrete_cmap(13, 'RdBu_r'),fmax=40,baseline=(-3,-1),mode="logratio")
raw_signal.plot(block=True)