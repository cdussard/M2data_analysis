#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:54:20 2021

@author: claire.dussard
"""

channelsSansFz = ['Fp1', 'Fp2', 'F7', 'F3','F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10','T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5','CP1','CP2','CP6','TP10','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2','HEOG','VEOG']

channelsAvecFz = ['Fp1', 'Fp2', 'F7', 'F3','Fz','F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10','T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5','CP1','CP2','CP6','TP10','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2','HEOG','VEOG']

for signal in listeFilteredICA[0:2]:
    bonOrdreEEG=signal.copy().reorder_channels(channelsSansFz)
    bonOrdreEEG.plot(block=True)
    
    
my_cmap = discrete_cmap(13, 'RdBu_r')
av_power_pendule.plot_topomap(tmin=1.5,tmax=26.8,fmin=8,fmax=30,vmin=-0.26,vmax=0.26,cmap=my_cmap)
av_power_main.plot_topomap(tmin=1.5,tmax=26.8,fmin=8,fmax=30,vmin=-0.26,vmax=0.26,cmap=my_cmap)
av_power_mainIllusion.plot_topomap(tmin=1.5,tmax=26.8,fmin=8,fmax=30,vmin=-0.26,vmax=0.26,cmap=my_cmap)
raw_signal.plot(block=True)

av_power_pendule_noBL = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule_noBaseline-tfr.h5")[0]
av_power_main_noBL = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main_noBaseline-tfr.h5")[0] 
av_power_mainIllusion_noBL = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion_noBaseline-tfr.h5")[0] 


av_power_pendule_noBL.plot_topomap(tmin=1.5,tmax=26.8,fmin=8,fmax=30,vmin=-0.26,vmax=0.26,cmap=my_cmap,mode=mode,baseline=bl)
av_power_main_noBL.plot_topomap(tmin=1.5,tmax=26.8,fmin=8,fmax=30,vmin=-0.26,vmax=0.26,cmap=my_cmap,mode=mode,baseline=bl)
av_power_mainIllusion_noBL.plot_topomap(tmin=1.5,tmax=26.8,fmin=8,fmax=30,vmin=-0.26,vmax=0.26,cmap=my_cmap,mode=mode,baseline=bl)
raw_signal.plot(block=True)


my_cmap = discrete_cmap(13, 'RdBu_r')
mode = "percent"
bl = (-3,-1)
# av_power_pendule_noBL.apply_baseline(mode=mode,baseline=bl)
# av_power_main_noBL.apply_baseline(mode=mode,baseline=bl)
# av_power_mainIllusion_noBL.apply_baseline(mode=mode,baseline=bl)

for data,leg in zip([av_power_pendule_noBL,av_power_main_noBL,av_power_mainIllusion_noBL],["pendulum","hand","hand+vibrations"]):
    data = data.copy()
    #data.apply_baseline(mode=mode,baseline=bl)
    data.crop(fmin=15,fmax=30,tmin=-3,tmax=26.8)
    plt.plot(data.times,np.median(data.data[11],axis=0),label=leg)
    plt.legend()
raw_signal.plot(block=True)



