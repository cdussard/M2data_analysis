#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:13:23 2021

@author: claire.dussard
"""
#try to compute a laplacian channel via raw data
signal = listeFiltered[0]
signal.describe()#sort un table avec min max median Q1 Q3
data_C3 = signal.get_data(picks=['C3'])
data_CP1 = signal.get_data(picks=['CP1'])
data_CP5 = signal.get_data(picks=['CP5'])
data_FC1 = signal.get_data(picks=['FC1'])
data_FC5 = signal.get_data(picks=['FC5'])


laplacianC3_data = data_C3 - 1/4*(data_CP1+data_CP5+data_FC1+data_FC5)

info_laplacian = mne.create_info(["laplacian_c3"], 1000, ch_types='eeg', verbose=None)
rawLaplacian_C3 = mne.io.RawArray(laplacianC3_data,info_laplacian)

rawLaplacian_C3.plot(block=True)

#via epochs
epochs = tousEpochsSujets_test[0:10]
dataEpoch_C3 = epochs.get_data(picks=['C3'])
dataEpoch_CP1 = epochs.get_data(picks=['CP1'])
dataEpoch_CP5 = epochs.get_data(picks=['CP5'])
dataEpoch_FC1 = epochs.get_data(picks=['FC1'])
dataEpoch_FC5 = epochs.get_data(picks=['FC5'])

laplacianC3_Epochdata = dataEpoch_C3 - 1/4*(dataEpoch_CP1+dataEpoch_CP5+dataEpoch_FC1+dataEpoch_FC5)

epochLaplacian_C3 = mne.EpochsArray(laplacianC3_Epochdata, info_laplacian)

powerLaplacian, itcLaplacian = mne.time_frequency.tfr_morlet(epochLaplacian_C3, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                           return_itc=True, decim=3, n_jobs=1)#premier sujet

powerLaplacian.plot_topomap(baseline=(dureePreBaseline,valeurPostBaseline),mode=mode,fmin = 4,fmax=8,tmin=0.0,vmin=-0.15,vmax=0.15)#theta
powerLaplacian.plot_topomap(baseline=(dureePreBaseline,valeurPostBaseline),mode=mode,fmin = 8,fmax=13,tmin=0.0,vmin=-0.26,vmax=0.26)#alpha #mettre une echelle commune
powerLaplacian.plot_topomap(baseline=(dureePreBaseline,valeurPostBaseline),mode=mode,fmin = 13,fmax=20,tmin=0.0,vmin=-0.15,vmax=0.15)#low beta
powerLaplacian.plot_topomap(baseline=(dureePreBaseline,valeurPostBaseline),mode=mode,fmin =20,fmax=30,tmin=0.0,vmin=-0.15,vmax=0.15)


# #=============== POWER DATA ==============
# av_power_main
# av_power_mainIllusion_C3 = av_power_mainIllusion.copy().pick_channels(['C3'])
# av_power_mainIllusion_CP1 = av_power_mainIllusion.copy().pick_channels(['CP1'])
# av_power_mainIllusion_CP5 = av_power_mainIllusion.copy().pick_channels(['CP5'])
# av_power_mainIllusion_FC1 = av_power_mainIllusion.copy().pick_channels(['FC1'])
# av_power_mainIllusion_FC5 = av_power_mainIllusion.copy().pick_channels(['FC5'])


# av_power_mainIllusion_laplacian = av_power_mainIllusion_C3 - 1/4 * av_power_mainIllusion_CP1 #ne marche pas
i = 0
liste_tfr_mainIllusion_C3_moins_Laplacien =  []
liste_tfr_main_C3_moins_Laplacien =  []
liste_tfr_pendule_C3_moins_Laplacien =  []
for epochSujet in EpochDataPendule:
    print(i)
    epochs = epochSujet #sujet i
    dataEpoch_C3 = epochs.get_data(picks=['C3'])
    dataEpoch_CP1 = epochs.get_data(picks=['CP1'])
    dataEpoch_CP5 = epochs.get_data(picks=['CP5'])
    dataEpoch_FC1 = epochs.get_data(picks=['FC1'])
    dataEpoch_FC5 = epochs.get_data(picks=['FC5'])
    
    laplacianC3_Epochdata = dataEpoch_C3 - 1/4*(dataEpoch_CP1+dataEpoch_CP5+dataEpoch_FC1+dataEpoch_FC5)
    
    info_laplacian = mne.create_info(["laplacian_c3"], 1000, ch_types='eeg', verbose=None)
    
    time_start = epochSujet._times_readonly[0]
    epochLaplacian_C3 = mne.EpochsArray(laplacianC3_Epochdata, info_laplacian,tmin=time_start)
    freqs = np.arange(3, 85, 1)
    n_cycles = freqs 
    powerLaplacian, itcLaplacian = mne.time_frequency.tfr_morlet(epochLaplacian_C3, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                               return_itc=True, decim=3, n_jobs=1)#premier sujet
    
    powerC3 = mne.time_frequency.tfr_morlet(epochs.copy().pick_channels(["C3"]), freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                               return_itc=False, decim=3, n_jobs=1)#premier sujet
    
    print(powerC3)
    print(powerLaplacian)
    power_laplacian_moins_C3 = powerLaplacian - powerC3
    #powerLaplacian.plot(baseline = (-4,1),mode='logratio')
    #powerC3.plot(baseline = (-4,-1),mode='logratio')#faire baselines avant de soustraire ?
    #power_laplacian_moins_C3.plot(baseline = (-4,-1),mode='logratio',fmin = 8,fmax = 30)#si on a du bleu : meilleure desynchro en laplacien que C3 seul
    liste_tfr_pendule_C3_moins_Laplacien.append(power_laplacian_moins_C3)
    i += 1

raw_signal.plot(block=True)

#moyenner pour avoir donnees globales sujets
for tfr_C3Laplacien in liste_tfr_pendule_C3_moins_Laplacien:
    tfr_C3Laplacien.apply_baseline((-4,-1),'logratio')
    
liste_tfr_pendule_C3_moins_Laplacien.pop(5)
av_power_tfr_C3Laplacien= mne.grand_average(liste_tfr_pendule_C3_moins_Laplacien,drop_bads=False)
av_power_tfr_C3Laplacien.save("../AV_TFR/all_sujets/pendule_C3_moins_Laplacien-tfr.h5",overwrite=True)
av_power_tfr_C3Laplacien.plot(baseline = None,mode='logratio',fmin = 3,fmax = 40,cmap = my_cmap)
# #discrete colorbar
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['#315dbf', '#b8e6fe'], N=2)
#ça marche <3