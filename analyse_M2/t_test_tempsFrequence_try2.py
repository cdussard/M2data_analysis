# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 18:25:57 2022

@author: claire.dussard
"""

liste_power_sujets


# Parameters:

tfr_contrast_array = np.zeros([23,82 ,9000])

for i in range(len(liste_power_sujets)):
    tfr_contrast_array[i,:,:] = np.squeeze(liste_power_sujets[i].data[11])
# Cluster permutation test:
t_obs, clusters, cluster_pv, H0 = \
    mne.stats.permutation_cluster_1samp_test(tfr_contrast_array, out_type='mask',
                            n_permutations=50, threshold=None, tail=1)
# Creating a mask of all the clusters:

mask = np.zeros_like(t_obs,dtype=bool)

for cluster in clusters:
    mask[cluster] = 1

# TFR for the contrast over all subjects combined:
tfr_combined = tfr_morlet(evoked_combined, freqs=freqs,
                          n_cycles=n_cycles, decim=3, n_jobs=4,
                          return_itc=False, zero_mean=True)
# Plotting the TFR with the mask:
fig_tfr_combined = tfr_combined.plot(baseline = baseline,
    tmin=tmin, 
    tmax=tmax, 
    cmap=cmap, # 'jet'
    title='TFR: Contrast, Threshold = None',
    mask = mask,
    mask_style = mask_style, # 'both'
    mask_cmap = cmap, # 'jet'
    mask_alpha = 0.5)