#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 14:57:32 2022

@author: claire.dussard
"""
import mne
import numpy as np
#load previous data
EpochDataMain = load_data_postICA_postdropBad(rawPath_main_sujets,"")

EpochDataPendule = load_data_postICA_postdropBad(rawPath_pendule_sujets,"")

EpochDataMainIllusion = load_data_postICA_postdropBad(rawPath_mainIllusion_sujets,"")


#compute laplacien for first condition

def compute_laplacian(epoch_list): #refonte de la fonction sans avoir un dico en entree
    laplacian_list = list() # creation of the laplacian dictionary (2 laplacians for each subject)
    laplacian = {'C3':['CP5','CP1','FC5', 'FC1'],'C4':['FC2', 'FC6','CP2','CP6']}
    for epoch in epoch_list: #epoch dictionnary 
        
        
        new_epoch_C3 = 4*epoch.copy().pick_channels(["C3"]).get_data()
        laplacians = ['CP5','CP1','FC5', 'FC1']
        for i in range(len(laplacians)):
            new_epoch_C3 = new_epoch_C3-epoch.copy().pick_channels([laplacians[i]]).get_data()
        
        new_epoch_C4 = 4*epoch.copy().pick_channels(["C4"]).get_data()
        laplacians = ['FC2', 'FC6','CP2','CP6']
        for i in range(len(laplacians)):
            new_epoch_C4 = new_epoch_C4-epoch.copy().pick_channels([laplacians[i]]).get_data()
        
        data = np.array([new_epoch_C3,
                 new_epoch_C4])
        data = data.mean(axis=2)
        data = data.reshape((data.shape[1],data.shape[0],data.shape[2]))
        info = mne.create_info(ch_names=["C3","C4"], sfreq=epoch.info['sfreq'], ch_types='eeg')
        new_epoch = mne.EpochsArray(data=data, info=info, verbose=0,tmin=-5)               
        laplacian_list.append(new_epoch)
    return laplacian_list

epochC3_laplacien_main = compute_laplacian(EpochDataMain)
epochC3_laplacien_mainIllusion = compute_laplacian(EpochDataMainIllusion)
epochC3_laplacien_pendule = compute_laplacian(EpochDataPendule)


montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in epochC3_laplacien_main:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
for epochs in epochC3_laplacien_mainIllusion:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
for epochs in epochC3_laplacien_pendule:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
        
        
liste_power_sujets = []
freqs = np.arange(3, 50, 1)
n_cycles = freqs 
i = 0
#EpochData = epochC3_laplacien_mainIllusion
#EpochData = epochC3_laplacien_main
EpochData = epochC3_laplacien_pendule

for epochs_sujet in EpochData:
    print("========================\nsujet"+str(i))
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...")
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False)
    print("computing power...")
    liste_power_sujets.append(power_sujet)
    i += 1



#on pourrait tout encapsuler dans une fonction avec un param normalisation par seuil ou baseline

liste_power_sujets_main = liste_power_sujets
save_tfr_data(liste_power_sujets_main,rawPath_main_sujets,"C3C4laplacien")

liste_power_sujets_pendule = liste_power_sujets
save_tfr_data(liste_power_sujets_pendule,rawPath_pendule_sujets,"C3C4laplacien")

liste_power_sujets_mainIllusion = liste_power_sujets
save_tfr_data(liste_power_sujets_mainIllusion,rawPath_mainIllusion_sujets,"C3C4laplacien")

dureePreBaseline = 3 #3
dureePreBaseline = - dureePreBaseline
dureeBaseline = 2.0 #2.0
valeurPostBaseline = dureePreBaseline + dureeBaseline

baseline = (dureePreBaseline, valeurPostBaseline)
for tfr in liste_power_sujets_main:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
for tfr in liste_power_sujets_mainIllusion:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
for tfr in liste_power_sujets_pendule:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)

av_power_main_C3C4laplacien = mne.grand_average(liste_power_sujets_main,interpolate_bads=True)
av_power_mainIllusion_C3C4laplacien = mne.grand_average(liste_power_sujets_mainIllusion,interpolate_bads=True)
av_power_pendule_C3C4laplacien = mne.grand_average(liste_power_sujets_pendule,interpolate_bads=True)


av_power_main_C3C4laplacien.save("../AV_TFR/all_sujets/main_C3C4laplacien-tfr.h5",overwrite=True)
av_power_mainIllusion_C3C4laplacien.save("../AV_TFR/all_sujets/mainIllusion_C3C4laplacien-tfr.h5",overwrite=True)
av_power_pendule_C3C4laplacien.save("../AV_TFR/all_sujets/pendule_C3C4laplacien-tfr.h5",overwrite=True)

av_power_pendule_C3C4laplacien.plot_topo(fmax=35)#cliquer sur C3,Cz,C4 et sauver graphes
raw_signal.plot(block=True)


av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main_C3C4laplacien-tfr.h5")[0]#mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0]#mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main_C3C4laplacien-tfr.h5")[0]
av_power_mainIllusion = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion_C3C4laplacien-tfr.h5")[0]#mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0] #mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion_C3C4laplacien-tfr.h5")[0]

av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule_C3C4laplacien-tfr.h5")[0]#mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]#

# ==========test the effect of laplacien==========
av_power_main.plot(picks=["C3"],fmax=40,vmin=-0.3,vmax=0.3)
av_power_mainIllusion.plot(picks=["C3"],fmax=40,vmin=-0.3,vmax=0.3)
av_power_pendule.plot(picks=["C3"],fmax=40,vmin=-0.3,vmax=0.3)#vmin=-0.5,vmax=0.5)
raw_signal.plot(block=True)
#compute map differences
diff_main = av_power_main.pick_channels(["C3","C4" ]) - av_power_main_C3C4laplacien
diff_mainIllusion = av_power_mainIllusion.pick_channels(["C3","C4" ]) - av_power_mainIllusion_C3C4laplacien
diff_pendule = av_power_pendule.pick_channels(["C3","C4" ]) - av_power_pendule_C3C4laplacien

diff_main.plot(picks=["C3","C4"],fmax=40,vmin=-0.35,vmax=0.35)
diff_mainIllusion.plot(picks=["C3","C4"],fmax=40,vmin=-0.35,vmax=0.35)
diff_pendule.plot(picks=["C3","C4"],fmax=40,vmin=-0.35,vmax=0.35)
raw_signal.plot(block=True)



#=========== meme chose sans la baseline ===================
#read les data laplacien
liste_power_sujets_mainIllusion = liste_power_sujets


for i in range(23):
    seuil = float(seuils_sujets["seuil_min_mvt"][i])
    print(seuil)
    mainIllusion_seuil = liste_power_sujets_mainIllusion[i].data/seuil
    liste_power_sujets_mainIllusion[i].data = mainIllusion_seuil

av_power_mainIllusion_C3laplacien_seuil = mne.grand_average(liste_power_sujets_mainIllusion,interpolate_bads=True)
av_power_mainIllusion_C3laplacien_seuil.save("../AV_TFR/all_sujets/mainIllusion_C3C4laplacien_noBL_seuil-tfr.h5",overwrite=True)
av_power_mainIllusion_C3laplacien_seuil.plot(picks=["C3"],fmin=3,fmax=40,vmin=0,vmax=0.9e-10)
raw_signal.plot(block=True)

fig,axes = plt.subplots()
av_power_mainIllusion_C3laplacien_seuil.plot(picks="C3",fmin=3,fmax=40,vmin=0,vmax=0.7e-10,axes=axes)
axes.axvline(2, color='green', linestyle='--')
axes.axvline(6.5, color='black', linestyle='--')
axes.axvline(8.3, color='black', linestyle='--')
axes.axvline(12.7, color='black', linestyle='--')
axes.axvline(14.5, color='black', linestyle='--')
axes.axvline(18.92, color='black', linestyle='--')
axes.axvline(20.72, color='black', linestyle='--')
axes.axvline(25.22, color='black', linestyle='--')
axes.axvline(27.02, color='black', linestyle='--')
axes.axvline(26.9, color='green', linestyle='--')
plt.show()
fig
raw_signal.plot(block=True)
#meme chose avec main
liste_power_sujets_main = liste_power_sujets
for i in range(23):
    seuil = float(seuils_sujets["seuil_min_mvt"][i])
    print(seuil)
    main_seuil = liste_power_sujets_main[i].data/seuil
    liste_power_sujets_main[i].data = main_seuil

av_power_main_C3laplacien_seuil = mne.grand_average(liste_power_sujets_main,interpolate_bads=True)
av_power_main_C3laplacien_seuil.save("../AV_TFR/all_sujets/main_C3C4laplacien_noBL_seuil-tfr.h5",overwrite=True)
fig,axes = plt.subplots()
av_power_main_C3laplacien_seuil.plot(picks="C3",fmin=3,fmax=40,vmin=0,vmax=0.7e-10,axes=axes)
axes.axvline(2, color='green', linestyle='--')
axes.axvline(6.5, color='black', linestyle='--')
axes.axvline(8.3, color='black', linestyle='--')
axes.axvline(12.7, color='black', linestyle='--')
axes.axvline(14.5, color='black', linestyle='--')
axes.axvline(18.92, color='black', linestyle='--')
axes.axvline(20.72, color='black', linestyle='--')
axes.axvline(25.22, color='black', linestyle='--')
axes.axvline(27.02, color='black', linestyle='--')
axes.axvline(26.9, color='green', linestyle='--')
plt.show()
fig
raw_signal.plot(block=True)

#meme chose avec pendule
liste_power_sujets_pendule = liste_power_sujets

for i in range(23):
    seuil = float(seuils_sujets["seuil_min_mvt"][i])
    print(seuil)
    pendule_seuil = liste_power_sujets_pendule[i].data/seuil
    liste_power_sujets_pendule[i].data = pendule_seuil

av_power_pendule_C3laplacien_seuil = mne.grand_average(liste_power_sujets_pendule,interpolate_bads=True)
av_power_pendule_C3laplacien_seuil.save("../AV_TFR/all_sujets/pendule_C3C4laplacien_noBL_seuil-tfr.h5",overwrite=True)

fig,axes = plt.subplots()
av_power_pendule_C3laplacien_seuil.plot(picks="C3",fmin=3,fmax=40,vmin=0,vmax=0.7e-10,axes=axes)
axes.axvline(2, color='green', linestyle='--')
axes.axvline(6.5, color='black', linestyle='--')
axes.axvline(8.3, color='black', linestyle='--')
axes.axvline(12.7, color='black', linestyle='--')
axes.axvline(14.5, color='black', linestyle='--')
axes.axvline(18.92, color='black', linestyle='--')
axes.axvline(20.72, color='black', linestyle='--')
axes.axvline(25.22, color='black', linestyle='--')
axes.axvline(27.02, color='black', linestyle='--')
axes.axvline(26.9, color='green', linestyle='--')
plt.show()
fig
raw_signal.plot(block=True)
