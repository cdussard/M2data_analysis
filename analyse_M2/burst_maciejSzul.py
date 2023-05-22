# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:58:37 2022

@author: claire.dussard
"""

from functions.functions_burst import extract_bursts

EpochDataMain = load_data_postICA_postdropBad_windows(rawPath_main_sujets[0:1],"",True)
liste_tfr_main = load_tfr_data_windows(rawPath_main_sujets[0:1],"allTrials",True)

bursts = extract_bursts(
    raw_trials, tf, times, search_freqs, 
    band_lims, aperiodic_spectrum, sfreq, w_size=.26
)

bursts_single_trial = extract_bursts_single_trial(
    raw_trial, tf, times, search_freqs, 
    band_lims, aperiodic_spectrum, sfreq, w_size=.26
)

fg = plot_foofGroup_data(liste_tfr_main[0],[13,35],0,25,True)
liste_ap_fits = list()
for i in range(10):
    ap_fit = fg.get_fooof(ind=0, regenerate=True)._ap_fit
    liste_ap_fits.append(ap_fit)


bursts = extract_bursts_single_trial(
    EpochDataMain[0], liste_tfr_main[0][0], EpochDataMain[0].times, liste_tfr_main[0].freqs, 
    [13,35], liste_ap_fits[0], 1000, w_size=.26
)


