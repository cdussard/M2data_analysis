#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:17:31 2021

@author: claire.dussard
"""

#illustrate how several artefacts look in the files & in TFR

#==========================================vagues basse fréquence==========================================
raw = mne.io.read_raw_brainvision(liste_rawPathMain[4],preload=True,eog=('HEOG', 'VEOG'))#sujet 6
recording_eeg = raw.drop_channels(['ECG' ,'ACC_X','ACC_Z','ACC_Y'])
recording_eeg.plot(block=True)
events = mne.events_from_annotations(raw)[0] 
epochsCibles = mne.Epochs(recording_eeg,events,{'Essai_main':3},tmin=-5,tmax = 31,baseline=None, preload=True)
epochsCibles.plot(block=True)
epochData_sujet_down = epochsCibles.resample(250., npad='auto')

epochData_sujet_down.drop_channels(['EMG'])
epochData_sujet_down.set_montage(montageEasyCap)
 
power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False)

epochVague = epochData_sujet_down[1]

power_vague = mne.time_frequency.tfr_morlet(epochVague, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                           return_itc=False, decim=3, n_jobs=1)

power_vague.plot(picks=['TP9','T7','TP10'],baseline=(-4,-1),mode='logratio',tmin = 0, tmax = 25.5)

power_vague.plot(picks=['TP9','T7','TP10'],baseline=None,mode='logratio',tmin = 0, tmax = 25.5)

#==========================================artefact électrode==========================================
raw = mne.io.read_raw_brainvision(liste_rawPathMain[5],preload=True,eog=('HEOG', 'VEOG'))#sujet 7

recording_eeg = raw.drop_channels(['ECG' ,'ACC_X','ACC_Z','ACC_Y'])
recording_eeg.plot(block=True)
events = mne.events_from_annotations(raw)[0] 
epochsCibles = mne.Epochs(recording_eeg,events,{'Essai_main':3},tmin=-5,tmax = 31,baseline=None, preload=True)
epochsCibles.plot(block=True)
epochData_sujet_down = epochsCibles.resample(250., npad='auto')

epochData_sujet_down.drop_channels(['EMG'])
epochData_sujet_down.set_montage(montageEasyCap)

epochArtefact = epochData_sujet_down[2]#TP10
power_artefact = mne.time_frequency.tfr_morlet(epochArtefact, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                           return_itc=False, decim=3, n_jobs=1)

power_artefact.plot(picks=['TP9','TP10'],baseline=(-4,-1),mode='logratio',tmin = 0, tmax = 25.5)#se voit mieux avec la BL

power_artefact.plot(picks=['TP9','TP10'],baseline=None,mode='logratio',tmin = 0, tmax = 25.5)

raw_signal.plot(block=True)#debloquer graphes

#==========================================coup de mâchoire==========================================
raw = mne.io.read_raw_brainvision(liste_rawPathMain[8],preload=True,eog=('HEOG', 'VEOG'))#sujet 6
recording_eeg = raw.drop_channels(['ECG' ,'ACC_X','ACC_Z','ACC_Y'])
recording_eeg.plot(block=True)
events = mne.events_from_annotations(raw)[0] 
epochsCibles = mne.Epochs(recording_eeg,events,{'Essai_main':3},tmin=-5,tmax = 31,baseline=None, preload=True)
epochsCibles.plot(block=True)
epochData_sujet_down = epochsCibles.resample(250., npad='auto')

epochData_sujet_down.drop_channels(['EMG'])
epochData_sujet_down.set_montage(montageEasyCap)
epochData_sujet_down.plot(block=True)

epochMachoire = epochData_sujet_down[8]

power_jaw = mne.time_frequency.tfr_morlet(epochMachoire, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                           return_itc=False, decim=3, n_jobs=1)

power_jaw.plot(picks=['FT10','Fp2','C3'],baseline=(-4,-1),mode='logratio',tmin = 0, tmax = 25.5)#se voit mieux avec la BL

power_jaw.plot(picks=['FT10','Fp2','C3'],baseline=None,mode='logratio',tmin = 0, tmax = 25.5)
#========================blink==========================================
raw = mne.io.read_raw_brainvision(liste_rawPathMain[5],preload=True,eog=('HEOG', 'VEOG'))#sujet 7

recording_eeg = raw.drop_channels(['ECG' ,'ACC_X','ACC_Z','ACC_Y'])
recording_eeg.plot(block=True)
events = mne.events_from_annotations(raw)[0] 
epochsCibles = mne.Epochs(recording_eeg,events,{'Essai_main':3},tmin=-5,tmax = 31,baseline=None, preload=True)
epochsCibles.plot(block=True)
epochData_sujet_down = epochsCibles.resample(250., npad='auto')

epochData_sujet_down.drop_channels(['EMG'])
epochData_sujet_down.set_montage(montageEasyCap)

epochBlinks = epochData_sujet_down[6]

power_blink = mne.time_frequency.tfr_morlet(epochBlinks, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                           return_itc=False, decim=3, n_jobs=1)

power_blink.plot(picks=['Fp1','Fp2','C3'],baseline=(-4,-1),mode='logratio',tmin = 0, tmax = 25.5)#se voit mieux avec la BL

power_blink.plot(picks=['Fp1','Fp2','C3'],baseline=None,mode='logratio',tmin = 0, tmax = 25.5)

#==========================================dermogramme==========================================