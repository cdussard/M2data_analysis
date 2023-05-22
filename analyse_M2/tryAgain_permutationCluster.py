# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:53:51 2022

@author: claire.dussard
"""
import os 
import seaborn as sns
import pathlib
import mne
#necessite d'avoir execute handleData_subject.py, et load_savedData avant 
import numpy as np 
# importer les fonctions definies par moi 
from handleData_subject import createSujetsData
from functions.load_savedData import *

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

# listeNumSujetsFinale.pop(1)
# listeNumSujetsFinale.pop(3)
# #try to do the permutation clustering
# listeSujetsPendule_C3 = []
# listeSujetsMain_C3 = []
# listeSujetsMainIllusion_C3 = []
# for sujet in listeNumSujetsFinale:
    
#     mat_p = scipy.io.loadmat('../MATLAB_DATA/'+sujet+'/'+sujet+'-penduletimePooled.mat')["data"]
#     listeSujetsPendule_C3.append(mat_p)
#     mat_m = scipy.io.loadmat('../MATLAB_DATA/'+sujet+'/'+sujet+'-maintimePooled.mat')["data"]
#     listeSujetsMain_C3.append(mat_m)
#     mat_mi = scipy.io.loadmat('../MATLAB_DATA/'+sujet+'/'+sujet+'-mainIllusiontimePooled.mat')["data"]
#     listeSujetsMainIllusion_C3.append(mat_mi)
# #elec x freq

pval = 0.001  #arbitrary
dfn = 3 - 1  #degrees of freedom numerator
dfd = 23 - 3  #degrees of freedom denominator
thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd) # F distribution

# from mne.channels import find_ch_adjacency
# #adjacency, ch_names = mne.channels.read_ch_adjacency("easycap32ch-avg")
# #adjacency, ch_names = find_ch_adjacency(EpochDataMain[0].drop_channels(["TP9","TP10","FT9","FT10"]).info, ch_type='eeg')
# #print(type(adjacency))  it's a sparse matrix!

# fig, ax = plt.subplots(figsize=(5, 4))
# ax.imshow(adjacency.toarray(), cmap='gray', origin='lower',
#           interpolation='nearest')
# ax.set_xlabel('{} Magnetometers'.format(len(ch_names)))
# ax.set_ylabel('{} Magnetometers'.format(len(ch_names)))
# ax.set_title('Between-sensor adjacency')
# fig.tight_layout()
# raw_signal.plot(block=True)


# #add freqs to the adjacency matrix
# from mne.stats import combine_adjacency
# tfr_adjacency = combine_adjacency(
#     len(freqValues),adjacency)

# #contre zero 
# F_obs, clusters, p_values, _  = mne.stats.permutation_cluster_test(listeSujetsPendule_C3,
#                                 threshold=thresh,n_permutations=10000)#liste d'elec x freq

# F_obs, clusters, p_values, _  = mne.stats.permutation_cluster_test(listeSujetsMain_C3,
#                                 threshold=thresh,n_permutations=10000)#liste d'elec x freq
# F_obs, clusters, p_values, _  = mne.stats.permutation_cluster_test(listeSujetsMainIllusion_C3,
#                                 threshold=thresh,n_permutations=10000)#liste d'elec x freq
# # #display results

# # main vs pendule
# F_obs, clusters, p_values, _  = mne.stats.permutation_cluster_test([listeSujetsPendule_C3,listeSujetsMain_C3],
#                                 threshold=thresh,n_permutations=10000)#liste d'elec x freq


av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0]
av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0]
av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]

sensor_adjacency, ch_names = mne.channels.read_ch_adjacency("easycapM1")

# However, we need to subselect the channels we are actually using
use_idx = [ch_names.index(ch_name)
           for ch_name in av_power_main.ch_names]
sensor_adjacency = sensor_adjacency[use_idx][:, use_idx]

# Our sensor adjacency matrix is of shape n_chs × n_chs
assert sensor_adjacency.shape == \
    (len(av_power_main.ch_names), len(av_power_main.ch_names))

# Now we need to prepare adjacency information for the time-frequency
# plane. For that, we use "combine_adjacency", and pass dimensions
# as in the data we want to test (excluding observations). Here:
# channels × frequencies × times
assert av_power_main.data.shape == (
    len(av_power_main), len(av_power_main.ch_names),
    len(av_power_main.freqs), len(av_power_main.times))
adjacency = mne.stats.combine_adjacency(
    sensor_adjacency, len(av_power_main.freqs), len(av_power_main.times))

# The overall adjacency we end up with is a square matrix with each
# dimension matching the data size (excluding observations) in an
# "unrolled" format, so: len(channels × frequencies × times)
assert adjacency.shape[0] == adjacency.shape[1] == \
    len(av_power_main.ch_names) * len(av_power_main.freqs) * len(av_power_main.times)



from mne.stats import permutation_cluster_1samp_test
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_1samp_test(av_power_main.data, n_permutations=500,
                                   threshold=thresh, tail=0,
                                   adjacency=adjacency,
                                   out_type='mask', verbose=True)


