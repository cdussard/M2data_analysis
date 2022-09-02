# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:20:51 2022

@author: claire.dussard
"""

import fooof
import mne
import numpy as np
from fooof import FOOOF


# Initialize a FOOOF object
fm = FOOOF()

#load data
# Import the FOOOF object
from fooof import FOOOF


def plot_foof_data(power_object,freq_range):
    power_object.pick_channels(["C3"])
    power_object.crop(tmin=1.5,tmax=25.5,fmin=min(freq_range),fmax=max(freq_range))
    print(power_object)
    freqs = power_object.freqs
    test = np.mean(power_object.data,axis=2)
    test = np.mean(test,axis=0)
    print(len(test)==len(freqs))
    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    fm = FOOOF()
    fm.fit(freqs, test, freq_range)
    #fm.report(freqs, test, freq_range)
    #fm.print_results()
    fm.plot()
    
num_sujet = 6
power_pendule = load_tfr_data_windows(liste_rawPathPendule[num_sujet:num_sujet+1],"",True)[0]
power_main = load_tfr_data_windows(liste_rawPathMain[num_sujet:num_sujet+1],"",True)[0]
power_mainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion[num_sujet:num_sujet+1],"",True)[0]


freq_range = [3, 35]
plot_foof_data(power_pendule,freq_range)
plot_foof_data(power_main,freq_range)
plot_foof_data(power_mainIllusion,freq_range)

