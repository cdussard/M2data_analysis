# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:52:54 2023

@author: claire.dussard
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import os
import pathlib

os.chdir("../../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

suj = 0 
path_suj_NFsess = [["ParkPitie_2019_11_28_LOp_GBMOV_M3_ON_MI_SIT_001.vhdr", 
                   "ParkPitie_2019_11_28_LOp_GBMOV_M3_ON_MI_SIT_002.vhdr",
                   "ParkPitie_2019_11_28_LOp_GBMOV_M3_ON_MI_SIT_003.vhdr",
                   "ParkPitie_2019_11_28_LOp_GBMOV_M3_ON_MI_SIT_004.vhdr",
                   "ParkPitie_2019_11_28_LOp_GBMOV_M3_ON_MI_SIT_005.vhdr",
                  "ParkPitie_2019_11_28_LOp_GBMOV_M3_ON_MI_SIT_006.vhdr",
                  "ParkPitie_2019_11_28_LOp_GBMOV_M3_OFF_BLEO_SIT_001.vhdr"
                 
                   
]
            ]

n_suj = ["LOUPH38","BENMO28"]
matplotlib.use('Qt5Agg')

n_trial = 0
tmin = 5
tmax = 40
fmin = 4
fmax = 60
data_rest = mne.io.read_raw_brainvision(n_suj[suj]+"/"+path_suj_NFsess[suj][-1],preload=True,eog=('HEOG', 'VEOG'))#,misc=('EMG'))#AJOUT DU MISC EMG
pass_eeg_rest = data_rest.filter(0.1,250, method='iir', iir_params=dict(ftype='butter', order=4))
filtered_rest = pass_eeg_rest.notch_filter(freqs=[50,100], filter_length='auto',phase='zero')
data_rest = filtered_rest
data_C3_rest = data_rest.get_data(picks=['C3'])
info_C3 = mne.create_info(["C3"], 1000, ch_types='eeg', verbose=None)
rawmoy_C3_rest = mne.io.RawArray(data_C3_rest,info_C3)
for n_trial in range(2):#len(path_suj_NFsess[suj])):
    data = mne.io.read_raw_brainvision(n_suj[suj]+"/"+path_suj_NFsess[suj][n_trial],preload=True,eog=('HEOG', 'VEOG'))#,misc=('EMG'))#AJOUT DU MISC EMG
    pass_eeg = data.filter(0.1,250, method='iir', iir_params=dict(ftype='butter', order=4))
    filtered = pass_eeg.notch_filter(freqs=[50,100], filter_length='auto',phase='zero')
    data = filtered
    data_C3 = data.get_data(picks=['C3'])

    fig, axs = plt.subplots(2)
    info_C3 = mne.create_info(["C3"], 1000, ch_types='eeg', verbose=None)
    rawmoy_C3 = mne.io.RawArray(data_C3,info_C3)
    mne.viz.plot_raw_psd(rawmoy_C3_rest,ax=axs[0],fmin=fmin,fmax=fmax,tmin = tmin,tmax=tmax)
    mne.viz.plot_raw_psd(rawmoy_C3,ax=axs[1],fmin=fmin,fmax=fmax,tmin = tmin,tmax=tmax)

    for ax in axs:
        for freq in [8,13,21,30,35]:
            ax.axvline(x=freq, color='black', linestyle='--', linewidth=0.5)