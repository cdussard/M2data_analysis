#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:05:42 2022

@author: claire.dussard
"""

#compute statistics
#===============================================================================================================================
#set up ANOVA
# #get the data
# mne.stats.f_oneway()

# conditions_sujet_power = list()
# list.append(liste_power_mainIllusion)
from functions.load_savedData import *

from handleData_subject import createSujetsData

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../..")
lustre_data_dir = "cenir/analyse/meeg/BETAPARK/_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

from functions.load_savedData import *

liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)



#================================================================================================================
#create ANOVA table
tableauANOVA = np.zeros(shape=(23,3))
tableauANOVAmediane = np.zeros(shape=(23,3))
for i in range(23):
    pendule_2dim = liste_tfr_pendule[i].data.mean(axis=0)#faire pour autres aussi
    main_2dim = liste_tfr_main[i].data.mean(axis=0)#faire pour autres aussi
    mainI_2dim = liste_tfrMainIllusion[i].data.mean(axis=0)#faire pour autres aussi
    #pool power
    #8-30Hz
    powerOverTime8_30Hz_pendule = np.mean(pendule_2dim,axis=0)
    powerOverTime8_30Hz_main = np.mean(main_2dim,axis=0)
    powerOverTime8_30Hz_mainI = np.mean(mainI_2dim,axis=0)


    C3_mov_pendule = computeMovingAverage(powerOverTime8_30Hz_pendule,24)#a sauvegarder
    C3_mov_main = computeMovingAverage(powerOverTime8_30Hz_main,24)
    C3_mov_mainIllusion = computeMovingAverage(powerOverTime8_30Hz_mainI,24)
    #pool time
    valuePower8_30Hz_pendule = np.mean(C3_mov_pendule,axis=0) #pour le dernier sujet
    valuePower8_30Hz_main = np.mean(C3_mov_main,axis=0)
    valuePower8_30Hz_mainIllusion = np.mean(C3_mov_mainIllusion,axis=0)
    valuePower8_30Hz_pendule_med = np.median(C3_mov_pendule,axis=0) #pour le dernier sujet
    valuePower8_30Hz_main_med = np.median(C3_mov_main,axis=0)
    valuePower8_30Hz_mainIllusion_med = np.median(C3_mov_mainIllusion,axis=0)
    tableauANOVA[i][0] = valuePower8_30Hz_pendule
    tableauANOVA[i][1] = valuePower8_30Hz_main
    tableauANOVA[i][2] = valuePower8_30Hz_mainIllusion
    tableauANOVAmediane[i][0] = valuePower8_30Hz_pendule_med
    tableauANOVAmediane[i][1] = valuePower8_30Hz_main_med
    tableauANOVAmediane[i][2] = valuePower8_30Hz_mainIllusion_med



tableauANOVA_apresBaseline = tableauANOVA
pd.DataFrame(tableauANOVA_apresBaseline).to_csv("tableauANOVA_apresBaseline_C3_8-30Hz_movingAverage_seuilsBL.csv")
#tableauANOVA_avantBaseline = tableauANOVA
#pd.DataFrame(tableauANOVA_avantBaseline).to_csv("tableauANOVA_avantBaseline_C3_8-30Hz.csv")

pd.DataFrame(tableauANOVAmediane).to_csv("tableauANOVAmediane_C3_8-30Hz_movingAverage_seuilsBL.csv")

from statsmodels.stats.anova import AnovaRM
allERDmediane = [val for val in zip(tableauANOVAmediane[:,0],tableauANOVAmediane[:,1],tableauANOVAmediane[:,2])]
allERDmean = [val for val in zip(tableauANOVA_apresBaseline[:,0],tableauANOVA_apresBaseline[:,1],tableauANOVA_apresBaseline[:,2])]

allERDmediane = list(sum(allERDmediane,()))
allERDmean = list(sum(allERDmean,()))

df_mean = pd.DataFrame({'condition': np.tile([1, 2, 3],23),
                   'sujet': np.repeat([0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],3),
                      'ERD':allERDmean})
df_mediane = pd.DataFrame({'condition': np.tile([1, 2, 3],23),
                   'sujet': np.repeat([0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],3),
                      'ERD':allERDmediane})

anovaMediane = AnovaRM(data=df_mediane, depvar='ERD', subject='sujet', within=['condition']).fit()
anovaMean = AnovaRM(data=df_mean, depvar='ERD', subject='sujet', within=['condition']).fit()
print(anovaMediane)
print(anovaMean)
# import numpy as np
# from mne.stats import f_mway_rm

# data = np.random.random((23, 3, 9000))
# fval, pval = f_mway_rm(data, factor_levels=[3], effects='A')


#============================== CLUSTERING ==========================================
#need adjacency matrix
# raw_signal_eeg = EpochDataPendule[0].pick_types(eeg=True)
# # raw_signal_eeg = raw_signal.drop_channels(['ECG','EMG'])
# raw_signal_eeg.set_montage(montageEasyCap)
# mat = mne.channels.find_ch_adjacency(raw_signal_eeg.pick_types(eeg=True).info, 'eeg')


# mne.stats.f_mway_rm(data, factor_levels, effects='all', correction=False, return_pvals=True)

#============= essai du 3 janvier=========================
dureePreBaseline = 3 #3
dureePreBaseline = - dureePreBaseline
dureeBaseline = 2.0 #2.0
valeurPostBaseline = dureePreBaseline + dureeBaseline
baseline = (dureePreBaseline, valeurPostBaseline)

liste_tfr_main = load_tfr_data(rawPath_main_sujets,"")
for tfr in liste_tfr_main:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 29)
  
liste_tfr_mainIllusion = load_tfr_data(rawPath_mainIllusion_sujets,"")
for tfr in liste_tfr_mainIllusion:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 29)
    
liste_tfr_pendule = load_tfr_data(rawPath_pendule_sujets,"")
for tfr in liste_tfr_pendule:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 29)#modifie a 29 pour eviter confusion 23 sujets / 23 frequences
    
#=========shape the data for clustering=================
condition_main = np.empty((23,6001,22,28), dtype=float)
ini_shape = liste_tfr_main[0].data.shape
for i in range(condition_main.shape[0]):
    donnees_sujet = liste_tfr_main[i].data
    donnees_sujet_reshaped = np.reshape(donnees_sujet,(ini_shape[2],ini_shape[1],ini_shape[0]))
    condition_main[i] = donnees_sujet_reshaped

condition_mainIllusion = np.empty((23,6001,22,28), dtype=float)
ini_shape = liste_tfr_mainIllusion[0].data.shape
for i in range(condition_mainIllusion.shape[0]):
    donnees_sujet = liste_tfr_mainIllusion[i].data
    donnees_sujet_reshaped = np.reshape(donnees_sujet,(ini_shape[2],ini_shape[1],ini_shape[0]))
    condition_mainIllusion[i] = donnees_sujet_reshaped

condition_pendule = np.empty((23,6001,22,28), dtype=float)#a lire au lieu d'encoder en dur
ini_shape = liste_tfr_pendule[0].data.shape
for i in range(condition_pendule.shape[0]):
    donnees_sujet = liste_tfr_pendule[i].data
    donnees_sujet_reshaped = np.reshape(donnees_sujet,(ini_shape[2],ini_shape[1],ini_shape[0]))
    condition_pendule[i] = donnees_sujet_reshaped
#=======================save the shaped data=======================
np.save('../numpy_files/mainIllusion_8-30_23sujets.npy', condition_mainIllusion)
np.save('../numpy_files/mainIllusion_8-29_23sujets.npy', condition_mainIllusion)
#save the file for future use
np.save('../numpy_files/main_8-30_23sujets.npy', condition_main)
np.save('../numpy_files/main_8-29_23sujets.npy', condition_main)
#save the file for future use
np.save('../numpy_files/pendule_8-30_23sujets.npy', condition_pendule)
np.save('../numpy_files/pendule_8-29_23sujets.npy', condition_pendule)

# #=============== try code on forum https://mne.discourse.group/t/mne-stats-permutation-cluster-1samp-test/3530

# tfr_array_pendule = np.stack([tfr.data for tfr in liste_tfr_pendule], axis=0) #bof : mauvais ordre ?
#=========== load the shaped data====================================
condition_main = np.load('../numpy_files/main_8-30_22sujets.npy')
condition_pendule = np.load('../numpy_files/pendule_8-30_22sujets.npy')
condition_mainIllusion = np.load('../numpy_files/mainIllusion_8-30_22sujets.npy')    



#get the data
X = [condition_pendule,condition_main]#,condition_mainIllusion]

#find the adjacency matrix
import numpy as np
from scipy.sparse import diags
from mne.stats import combine_adjacency
n_times, n_freqs, n_chans = (6001, 23, 28)
chan_adj = diags([1., 1.], offsets=(-1, 1), shape=(n_chans, n_chans))
mat_adj = combine_adjacency(
    n_times,  # regular lattice adjacency for times
    np.zeros((n_freqs, n_freqs)),  # no adjacency between freq. bins
    chan_adj,  # custom matrix, or use mne.channels.find_ch_adjacency
    )  
#display it  $
# from mne.channels import find_ch_adjacency
# ch_names = liste_tfr_pendule[0].ch_names

# # read_M1_adj = mne.channels.read_ch_adjacency("../neighbor_electrodes/easycapM1_neighb.mat")

# adjacency, ch_names = find_ch_adjacency(liste_tfr_pendule[0].info, ch_type='eeg')

# print(type(adjacency))  # it's a sparse matrix!

plt.imshow(mat_adj.toarray(), cmap='gray', origin='lower',
           interpolation='nearest')
plt.xlabel('{} EEG'.format(len(ch_names)))
plt.ylabel('{} EEG'.format(len(ch_names)))
plt.title('Between-sensor adjacency')
#trop grande taille pour display


#================== define within subject t-test as stat=======================
from scipy.stats import ttest_rel

def ttest_rel_nop(*args):
    tvals, _ = ttest_rel(*args)
    return tvals
#======= set threshold ==============
from scipy import stats as stats
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., 23 - 1)#t_threshold = 2.074
#=====================================================================
T_obs, clusters, cluster_p_values, H0 = \
        mne.stats.permutation_cluster_test(X, n_permutations=25, adjacency = mat_adj,stat_fun=ttest_rel_nop,verbose=True,threshold=t_threshold)#out_type='mask'
  
save = T_obs, clusters, cluster_p_values, H0        
good_clusters = np.where(cluster_p_values < .05)[0] #all clusters >1.0 p value
print(good_clusters)

#=========== display the clusters
clu = save #two many dimensions when type = mask
print('Visualizing clusters.')
#============== try with a pooled frequency dimension ===================
#pool the 8-30Hz data
condition_main_freqpooled = np.empty((22,6001,28), dtype=float)
for i in range(condition_main_freqpooled.shape[0]):
    donnees_sujet = condition_main[i].data
    donnees_sujet_tp = np.mean(donnees_sujet,axis=1)#mean over time
    condition_main_freqpooled[i] = donnees_sujet_tp
    
condition_mainIllusion_freqpooled = np.empty((22,6001,28), dtype=float)
for i in range(condition_mainIllusion_freqpooled.shape[0]):
    donnees_sujet = condition_mainIllusion[i].data
    donnees_sujet_tp = np.mean(donnees_sujet,axis=1)#mean over time
    condition_mainIllusion_freqpooled[i] = donnees_sujet_tp

condition_pendule_freqpooled = np.empty((22,6001,28), dtype=float)
for i in range(condition_pendule_freqpooled.shape[0]):
    donnees_sujet = condition_pendule[i].data
    donnees_sujet_tp = np.mean(donnees_sujet,axis=1)#mean over time
    condition_pendule_freqpooled[i] = donnees_sujet_tp
    
#save the file for future use
np.save('../numpy_files/mainIllusion_8-30Pooled_22sujets.npy', condition_mainIllusion_freqpooled)
np.save('../numpy_files/main_8-30Pooled_22sujets.npy', condition_main_freqpooled)
np.save('../numpy_files/pendule_8-30Pooled_22sujets.npy', condition_pendule_freqpooled)

#================================================pool 8-13Hz ALPHA frequencies======================================

liste_tfr_main = load_tfr_data(rawPath_main_sujets[0:1],"")
for tfr in liste_tfr_main:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 13)
  
liste_tfr_mainIllusion = load_tfr_data(rawPath_mainIllusion_sujets,"")
for tfr in liste_tfr_mainIllusion:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 13)
    
liste_tfr_pendule = load_tfr_data(rawPath_pendule_sujets,"")
for tfr in liste_tfr_pendule:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 13)
#================================
#shape the data 
condition_main_alphapooled = np.empty((22,6001,28), dtype=float)
ini_shape = liste_tfr_main[0].data.shape
for i in range(condition_main_alphapooled.shape[0]):
    donnees_sujet = liste_tfr_main[i].data
    donnees_sujet_tp = np.mean(donnees_sujet,axis=1)#mean over time
    donnees_sujet_reshaped = np.reshape(donnees_sujet_tp,(ini_shape[2],ini_shape[0]))
    condition_main_alphapooled[i] = donnees_sujet_reshaped
    
condition_mainIllusion_alphapooled = np.empty((22,6001,28), dtype=float)
ini_shape = liste_tfr_mainIllusion[0].data.shape
for i in range(condition_mainIllusion_alphapooled.shape[0]):
    donnees_sujet = liste_tfr_mainIllusion[i].data
    donnees_sujet_tp = np.mean(donnees_sujet,axis=1)#mean over time
    donnees_sujet_reshaped = np.reshape(donnees_sujet_tp,(ini_shape[2],ini_shape[0]))
    condition_mainIllusion_alphapooled[i] = donnees_sujet_reshaped

condition_pendule_alphapooled = np.empty((22,6001,28), dtype=float)
ini_shape = liste_tfr_pendule[0].data.shape
for i in range(condition_pendule_alphapooled.shape[0]):
    donnees_sujet = liste_tfr_pendule[i].data
    donnees_sujet_tp = np.mean(donnees_sujet,axis=1)#mean over time
    donnees_sujet_reshaped = np.reshape(donnees_sujet_tp,(ini_shape[2],ini_shape[0]))
    condition_pendule_alphapooled[i] = donnees_sujet_reshaped
    
#save the file for future use
np.save('../numpy_files/main_8-13Pooled_22sujets.npy', condition_main_freqpooled)
np.save('../numpy_files/mainIllusion_8-13Pooled_22sujets.npy', condition_mainIllusion_freqpooled)
np.save('../numpy_files/pendule_8-13Pooled_22sujets.npy', condition_pendule_freqpooled)
#===============================

X_freqpooled = [condition_mainIllusion_freqpooled,condition_main_freqpooled]

X_alphapooled = [condition_main_alphapooled,condition_mainIllusion_alphapooled]#[condition_pendule_alphapooled,condition_main_alphapooled]

#mat adj
from scipy.sparse import diags
from mne.stats import combine_adjacency
n_times, n_chans = (6001, 28)
chan_adj = diags([1., 1.], offsets=(-1, 1), shape=(n_chans, n_chans))
mat_adj_freq_pooled = combine_adjacency(
    n_times,  # regular lattice adjacency for times,
    chan_adj,  # custom matrix, or use mne.channels.find_ch_adjacency
    )  

from scipy import stats as stats
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., 23 - 1)#t_threshold = 2.074

T_obs, clusters, cluster_p_values, H0 = \
        mne.stats.permutation_cluster_test(X_freqpooled, n_permutations=500, adjacency = mat_adj_freq_pooled,stat_fun=ttest_rel_nop,verbose=True,threshold=t_threshold)#out_type='mask'

save = T_obs, clusters, cluster_p_values, H0  
save2 = T_obs, clusters, cluster_p_values, H0  
save3 = T_obs, clusters, cluster_p_values, H0  
save4 =  T_obs, clusters, cluster_p_values, H0 #main vs pendule alpha 200 perm
save5 =  T_obs, clusters, cluster_p_values, H0 #main vs pendule alpha 500 perm
save6 =  T_obs, clusters, cluster_p_values, H0 #main vs mainI alpha 200 perm 
save7 =  T_obs, clusters, cluster_p_values, H0 #main vs mainI alpha 500 perm
save8 = T_obs, clusters, cluster_p_values, H0 #main vs mainI 8-30 200 perm
save9 = T_obs, clusters, cluster_p_values, H0 #main vs mainI 8-30 500 perm
good_clusters = np.where(cluster_p_values < .05)[0] #all clusters >1.0 p value
print(good_clusters)
#============= visualize the clusters

print('Visualizing clusters.')
clu = T_obs, clusters, cluster_p_values, H0
#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
fsave_vertices = [np.arange(14), np.arange(14)]
stc_all_cluster_vis = mne.stats.summarize_clusters_stc(clu, tstep=1/1000,
                                             vertices=fsave_vertices,
                                             subject='fsaverage') #No significant clusters available. Please adjust your threshold or check your statistical analysis.

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration
subjects_dir = op.join(data_path, 'subjects')
# blue blobs are for condition A != condition B
brain = stc_all_cluster_vis.plot('fsaverage', hemi='both',
                                 views='lateral', subjects_dir=subjects_dir,
                                 time_label='temporal extent (ms)',
                                 clim=dict(kind='value', lims=[0, 1, 40]))
#===========pool the time=====================================================
condition_main = np.load('../numpy_files/main_8-30_23sujets.npy')
condition_mainIllusion = np.load('../numpy_files/mainIllusion_8-30_23sujets.npy')
condition_pendule = np.load('../numpy_files/pendule_8-30_23sujets.npy')

condition_main_timepooled = np.empty((23,23,28), dtype=float)
ini_shape_timepooled = condition_main[0].data.shape
for i in range(condition_main_timepooled.shape[0]):
    donnees_sujet = condition_main[i].data
    donnees_sujet_tp = np.mean(donnees_sujet,axis=0)#mean over time
    condition_main_timepooled[i] = donnees_sujet_tp

    
condition_pendule_timepooled = np.empty((23,23,28), dtype=float)#a lire au lieu d'encoder en dur
ini_shape_timepooled = condition_pendule[0].data.shape
for i in range(condition_pendule_timepooled.shape[0]):
    donnees_sujet = condition_pendule[i].data
    donnees_sujet_tp = np.mean(donnees_sujet,axis=0)#mean over time
    condition_pendule_timepooled[i] = donnees_sujet_tp

condition_mainIllusion_timepooled = np.empty((23,23,28), dtype=float)
ini_shape_timepooled = condition_mainIllusion[0].data.shape
for i in range(condition_main_timepooled.shape[0]):
    donnees_sujet = condition_mainIllusion[i].data
    donnees_sujet_tp = np.mean(donnees_sujet,axis=0)#mean over time
    condition_mainIllusion_timepooled[i] = donnees_sujet_tp

X_tp = [condition_pendule_timepooled,condition_main_timepooled,condition_mainIllusion_timepooled]

pthresh = 0.05
f_thresh = mne.stats.f_threshold_mway_rm(n_subjects=23, factor_levels=[3], effects=["A"], pvalue=pthresh)

factor_levels=[3]
effects = ["A"]

from mne.stats import f_mway_rm
def stat_fun(*args):#repeated measures anova
    # get f-values only.
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]

T_obs, clusters, cluster_p_values, H0 = \
        mne.stats.permutation_cluster_test(X_tp,threshold=2.0, n_permutations=1000,stat_fun=stat_fun,tail=1)#threshold=f_thresh

res = f_mway_rm(np.swapaxes(X_tp, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=True)

from statsmodels.stats.anova import AnovaRM
AnovaRM(np.swapaxes(X_tp, 1, 0), 'rt', 'Sub_id', within=['cond'])

good_clusters = np.where(cluster_p_values < .05)[0]#all clusters >1.0 p value
print(good_clusters)

# #============= compute over mean of subjects A NE PAS FAIRE =========================
# # main_power = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")
# # main_power = main_power[0].crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)

# # mainIllusion_power = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")
# # mainIllusion_power = mainIllusion_power[0].crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)

# # pendule_power = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")
# # pendule_power = pendule_power[0].crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)

# # #reshape data
# # import matplotlib.pyplot as plt

# # condition_main_mean = np.empty((6001,23,28), dtype=float)
# # ini_shape = main_power.data.shape
# # donnees_sujet = main_power.data
# # donnees_sujet_reshaped = np.reshape(donnees_sujet,(ini_shape[2],ini_shape[1],ini_shape[0]))
# # condition_main_mean = donnees_sujet_reshaped

# # condition_mainIllusion_mean = np.empty((6001,23,28), dtype=float)
# # ini_shape = mainIllusion_power.data.shape
# # donnees_sujet = mainIllusion_power.data
# # donnees_sujet_reshaped = np.reshape(donnees_sujet,(ini_shape[2],ini_shape[1],ini_shape[0]))
# # condition_mainIllusion_mean = donnees_sujet_reshaped

# # condition_pendule_mean = np.empty((6001,23,28), dtype=float)
# # ini_shape = pendule_power.data.shape
# # donnees_sujet = pendule_power.data
# # donnees_sujet_reshaped = np.reshape(donnees_sujet,(ini_shape[2],ini_shape[1],ini_shape[0]))
# # condition_pendule_mean = donnees_sujet_reshaped

# # X_mean = [condition_pendule_mean,condition_main_mean,condition_mainIllusion_mean]

# # T_obs, clusters, cluster_p_values, H0 = \
# #         mne.stats.permutation_cluster_test(X_mean, n_permutations=100, tail=0)#, adjacency = mat[0])
        
# # T_obs, clusters, cluster_p_values, H0 = \
# #         mne.stats.permutation_cluster_test(X_mean, n_permutations=100, tail=0,out_type='mask')#, adjacency = mat[0])
        
        
# times = 1e3 * main_power.times   
# freqs = np.arange(8, 30, 1)
# plt.figure()
# #plt.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)

# plt.subplot(2, 1, 1)
# # Create new stats image with only significant clusters
# T_obs_plot = np.nan * np.ones_like(T_obs)
# for c, p_val in zip(clusters, cluster_p_values):
#     if p_val <= 0.05:
#         T_obs_plot[c] = T_obs[c]

# plt.imshow(T_obs,
#            extent=[times[0], times[-1], freqs[0], freqs[-1]],
#            aspect='auto', origin='lower', cmap='gray')
# plt.imshow(T_obs_plot,
#            extent=[times[0], times[-1], freqs[0], freqs[-1]],
#            aspect='auto', origin='lower', cmap='RdBu_r')

# plt.xlabel('Time (ms)')
# plt.ylabel('Frequency (Hz)')
# ch_name = '???'
# plt.title('Induced power (%s)' % ch_name)

# # ax2 = plt.subplot(2, 1, 2)
# # evoked_contrast = mne.combine_evoked([evoked_condition_1, evoked_condition_2],
# #                                      weights=[1, -1])
# # evoked_contrast.plot(axes=ax2, time_unit='s')

# plt.show()

# good_clusters = np.where(cluster_p_values < .05)[0]#all clusters >1.0 p value
# print(good_clusters)
# #je comprends rien 
# good_cluster_inds = np.where(cluster_p_values < 0.05)[0] #https://mne.discourse.group/t/topoplots-for-visualizing-clusters-from-permutation-cluster-1samp-test/2712/2

#=================F MAP==============================
condition_main_alphapooled = np.load('../numpy_files/main_8-13Pooled_22sujets.npy')
condition_pendule_alphapooled = np.load('../numpy_files/pendule_8-13Pooled_22sujets.npy')
condition_mainIllusion_alphapooled = np.load('../numpy_files/mainIllusion_8-13Pooled_22sujets.npy')    

# from mne.stats import permutation_t_test
# n_permutations = 1000
# condition_main_alphapooled.shape
# T0, p_values, H0 = permutation_t_test(condition_main_alphapooled, n_permutations, n_jobs=1)

# significant_sensors = picks[p_values <= 0.05]
# significant_sensors_names = [raw.ch_names[k] for k in significant_sensors]

def stat_fun(*args):#repeated measures anova
    # get f-values only.
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=return_pvals)[0]

pthresh = 0.05
f_thresh = mne.stats.f_threshold_mway_rm(n_subjects=23, factor_levels=3, effects=["A"], pvalue=pthresh)

#re try
condition_main = np.load('../numpy_files/main_8-30_23sujets.npy')
condition_pendule = np.load('../numpy_files/pendule_8-30_23sujets.npy')
condition_mainIllusion = np.load('../numpy_files/mainIllusion_8-30_23sujets.npy')    


stat,pval = scipy.stats.ttest_rel(condition_main,condition_pendule)
# ====retenter les permut
from scipy.stats import ttest_rel
X = [condition_main,condition_pendule]
def ttest_rel_nop(*args):
    tvals, _ = ttest_rel(*args)
    return tvals
#======= set threshold ==============
from scipy import stats as stats
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., 23 - 1)#t_threshold = 2.074
from scipy.sparse import diags
from mne.stats import combine_adjacency
n_times, n_freqs, n_chans = (6001, 23, 28)
chan_adj = diags([1., 1.], offsets=(-1, 1), shape=(n_chans, n_chans))
mat_adj = combine_adjacency(
    n_times,  # regular lattice adjacency for times
    np.zeros((n_freqs, n_freqs)),  # no adjacency between freq. bins
    chan_adj,  # custom matrix, or use mne.channels.find_ch_adjacency
    )  
#=====================================================================
T_obs, clusters, cluster_p_values, H0 = \
        mne.stats.permutation_cluster_test(X, n_permutations=500, adjacency = mat_adj,stat_fun=ttest_rel_nop,verbose=True,threshold=t_threshold)#out_type='mask'
  
good_clusters = np.where(cluster_p_values < .05)[0]# :')
#=============================================================================

liste_tfrMain = liste_power_sujets_main#load_tfr_data(rawPath_main_sujets,"")
liste_tfrPendule = liste_power_sujets_pendule# load_tfr_data(rawPath_pendule_sujets,"")
liste_tfrMainIllusion =  liste_power_sujets_mainIllusion#load_tfr_data(rawPath_mainIllusion_sujets,"")

for tfrmain,tfrpendule,tfrmainIllusion in zip(liste_tfrMain,liste_tfrMainIllusion,liste_tfrPendule):
    tfrmain.crop(fmin=3,fmax=30)
    tfrpendule.crop(fmin=3,fmax=30)
    tfrmainIllusion.crop(fmin=3,fmax=30)
#=============
n_conditions = 3
n_replications = len(liste_tfrMain) #// n_conditions
decim = 1
factor_levels = [3]  # number of levels in each factor
effects = 'A'  
freqs= tfrmain.freqs
n_freqs = len(freqs)
times = 1e3 * tfrmain.times[::decim]
n_times = len(times)

#get the data in the right format ======
epochs_power_m = np.empty((23,28,28,9000), dtype=float)
epochs_power_mi = np.empty((23,28,28,9000), dtype=float)
epochs_power_p = np.empty((23,28,28,9000), dtype=float)
epochs_power = list()

i = 0
for this_tfr in liste_tfrMain:
    this_power = this_tfr.data#[:, 11, :]  # C3 channel
    epochs_power_m[i]=this_power
    i = i +1
i = 0
for this_tfr in liste_tfrMainIllusion:
    this_power = this_tfr.data#[:, 11, :]  # C3 channel
    epochs_power_mi[i] = this_power
    i = i +1
i = 0
for this_tfr in liste_tfrPendule:
    this_power = this_tfr.data#[:, 11, :]  # C3 channel
    epochs_power_p[i] = this_power
    i = i +1
epochs_power.append(epochs_power_m)
epochs_power.append(epochs_power_mi)
epochs_power.append(epochs_power_p)
data = np.swapaxes(np.asarray(epochs_power), 1, 0)
n_conditions = 2
factor_levels = [2]
data = data.reshape(n_replications, n_conditions, n_freqs * n_times)

# so we have replications * conditions * observations:
print(data.shape)

#======== compute the test
import scipy
res = scipy.stats.ttest_rel(epochs_power_m,epochs_power_p)
fvals, pvals = f_mway_rm(data, factor_levels, effects=effects)
print(len(pvals[pvals<0.05]))
print(len(pvals[pvals<0.01]))

ch_name="C3"
effect_labels = ['condition']
plt.figure()
plt.imshow(fvals.reshape(28, 9000), cmap=plt.cm.gray, extent=[times[0],
               times[-1], freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
#fvals[pvals >= 0.5] = np.nan
fvals[fvals>3.3] = np.nan
pvals[pvals>0.05] = np.nan
plt.imshow(fvals.reshape(28, 9000), cmap='RdBu_r', extent=[times[0],
           times[-1], freqs[0], freqs[-1]], aspect='auto',
           origin='lower')
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title(r"Time-locked response for '%s' (%s)" % (effect_label, ch_name))
plt.show()

pthresh = 0.05
f_thresh = mne.stats.f_threshold_mway_rm(n_subjects=23, factor_levels=[3], effects=["A"], pvalue=pthresh)


#account for multiple tests
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]


epo_t_m = epochs_power[0][:,:,375:6375]
epo_m = epo_t_m.mean(axis=2)

epo_t_mi = epochs_power[1][:,:,375:6375]
epo_mi = epo_t_mi.mean(axis=2)

epo_t_p = epochs_power[2][:,:,375:6375]
epo_p = epo_t_p.mean(axis=2)

epochs_power = list()
epochs_power.append(epo_m)
epochs_power.append(epo_mi)
epochs_power.append(epo_p)

tail = 1  # f-test, so tail > 0
n_permutations = 20  # Save some time (the test won't be too sensitive ...)
T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
    epochs_power, stat_fun=stat_fun, threshold=f_thresh, tail=tail, n_jobs=1,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')

good_clusters = np.where(cluster_p_values < .05)[0]
T_obs_plot = T_obs.copy()
T_obs_plot[~clusters[np.squeeze(good_clusters)]] = np.nan

plt.figure()
for f_image, cmap in zip([T_obs, T_obs_plot], [plt.cm.gray, 'RdBu_r']):
    plt.imshow(f_image, cmap=cmap, extent=[times[0], times[-1],
               freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title("Time-locked response for 'modality by location' (%s)\n"
          " cluster-level corrected (p <= 0.05)" % ch_name)
plt.show()


#===========================================essai F test du 18/01 ====================================================================================
condition_main = np.load('../numpy_files/main_8-29_23sujets.npy')
condition_pendule = np.load('../numpy_files/pendule_8-29_23sujets.npy')
condition_mainIllusion = np.load('../numpy_files/mainIllusion_8-29_23sujets.npy')   

condition_main_timepooled_f = condition_main.mean(dtype=np.float64,axis=1)
condition_main_timepooled = condition_main.mean(axis=1)
condition_main_timepooled_freqpooled = condition_main_timepooled.mean(axis=1)

condition_pendule_timepooled = condition_pendule.mean(axis=1)
condition_pendule_timepooled_freqpooled = condition_pendule_timepooled.mean(axis=1)

condition_mainIllusion_timepooled = condition_mainIllusion.mean(axis=1)
condition_mainIllusion_timepooled_freqpooled = condition_mainIllusion_timepooled.mean(axis=1)

X_full_dim

from scipy.stats import ttest_rel
res = ttest_rel(condition_main_timepooled_freqpooled,condition_mainIllusion_timepooled_freqpooled,axis=1)#rien de signif sur 8-30Hz
print(res)
res = ttest_rel(condition_main_timepooled_freqpooled,condition_pendule_timepooled_freqpooled)#rien de signif sur 8-30Hz : les donnes sont bizarres
print(res)

diff = condition_main_timepooled_freqpooled - condition_pendule_timepooled_freqpooled
diff_C3 = diff[:,11]

X = [condition_main_timepooled_freqpooled,condition_pendule_timepooled_freqpooled,condition_mainIllusion_timepooled_freqpooled]

data = np.swapaxes(X, 1, 0)

fvals, pvals = mne.stats.f_mway_rm(data, [3], effects="A")

fvals_c, pvals_c = mne.stats.f_mway_rm(data, [3], effects="A",correction=True)#rien signif sur 8-30

#sur 8-13 Hz
condition_main_8_13 = condition_main.mean(axis=1)[:,0:5,:].mean(axis=1)
condition_pendule_8_13 = condition_pendule.mean(axis=1)[:,0:5,:].mean(axis=1)
condition_mainIllusion_8_13 = condition_mainIllusion.mean(axis=1)[:,0:5,:].mean(axis=1)
res = ttest_rel(condition_main_8_13,condition_pendule_8_13)
print(res)
X = [condition_main_8_13,condition_pendule_8_13,condition_mainIllusion_8_13]

data = np.swapaxes(X, 1, 0)
fvals_c, pvals_c = mne.stats.f_mway_rm(data, [3], effects="A",correction=True)
#sur 13-20Hz
condition_main_13_20 = condition_main.mean(axis=1)[:,6:12,:].mean(axis=1)
condition_pendule_13_20 = condition_pendule.mean(axis=1)[:,6:12,:].mean(axis=1)
condition_mainIllusion_13_20 = condition_mainIllusion.mean(axis=1)[:,6:12,:].mean(axis=1)
res = ttest_rel(condition_main_13_20,condition_pendule_13_20)
print(res)

X = [condition_main_13_20,condition_pendule_13_20,condition_mainIllusion_13_20]

data = np.swapaxes(X, 1, 0)
fvals_c, pvals_c = mne.stats.f_mway_rm(data, [3], effects="A",correction=True)
print(pvals_c)
#sur 20-30 Hz
condition_main_20_30 = condition_main.mean(axis=1)[:,12:,:].mean(axis=1)
condition_pendule_20_30 = condition_pendule.mean(axis=1)[:,12:,:].mean(axis=1)
condition_mainIllusion_20_30 = condition_mainIllusion.mean(axis=1)[:,12:,:].mean(axis=1)

res = ttest_rel(condition_main_20_30,condition_mainIllusion_20_30)
print(res)
X = [condition_main_20_30,condition_pendule_20_30,condition_mainIllusion_20_30]

data = np.swapaxes(X, 1, 0)
fvals_c, pvals_c = mne.stats.f_mway_rm(data, [3], effects="A",correction=True)
print(pvals_c)


#plot values
ch_names = ['Fp1', 'Fp2', 'F7', 'F3','Fz','F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5','CP1','CP2','CP6','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
raw_signal.drop_channels([#"FT9","FT10","TP9","TP10",'VEOG',
 'HEOG',
 'ECG',
 'EMG',
 'ACC_X',
 'ACC_Y',
 'ACC_Z'])
print(len(raw_signal.info.ch_names))
evoked = mne.EvokedArray(condition_main_8_13.mean(axis=0)[:, np.newaxis],
                         raw_signal.info, tmin=0.)
montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
evoked.set_montage(montageEasyCap)
# Extract mask and indices of active sensors in the layout
# stats_picks = mne.pick_channels(evoked.ch_names, significant_sensors_names)
# mask = p_values[:, np.newaxis] <= 0.05

evoked.plot_topomap(ch_type='eeg', times=[0], scalings=1,
                    time_format=None, cmap='Reds', #vmin=0., vmax=np.max,
                    units='p')#, cbar_fmt='-%0.1f', #mask=mask,
                    # size=3, show_names=lambda x: x[4:] + ' ' * 20,
                    # time_unit='s')

raw_signal.plot(block=True)