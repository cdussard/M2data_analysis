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


def plot_foof_data(power_object,freq_range,tmin,tmax,keepObject):
    if keepObject:
        power_object = power_object.copy()
    power_object.pick_channels(["C3"])
    power_object.crop(tmin=tmin,tmax=tmax,fmin=min(freq_range),fmax=max(freq_range))
    print(power_object)
    freqs = power_object.freqs
    test = np.mean(power_object.data,axis=2)
    test = np.mean(test,axis=0)
    print(len(test)==len(freqs))
    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    mean_peak = 2*(freqs[1]-freqs[0])
    fm = FOOOF(peak_width_limits=(mean_peak,12))
    fm.fit(freqs, test, freq_range)
    #fm.report(freqs, test, freq_range)
    #fm.print_results()
    fm.plot()
    
num_sujet = 0 #entre 0 et 22
power_pendule = load_tfr_data_windows(liste_rawPathPendule[num_sujet:num_sujet+1],"",True)[0]
power_main = load_tfr_data_windows(liste_rawPathMain[num_sujet:num_sujet+1],"",True)[0]
power_mainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion[num_sujet:num_sujet+1],"",True)[0]


freq_range = [3, 35]
tmin = 1.5
tmax = 3.5
plot_foof_data(power_pendule,freq_range,tmin,tmax)
plot_foof_data(power_main,freq_range,tmin,tmax,True)
plot_foof_data(power_mainIllusion,freq_range,tmin,tmax)

freq_range = [3, 35]
tmin = -3
tmax = -1
plot_foof_data(power_pendule,freq_range,tmin,tmax)
plot_foof_data(power_main,freq_range,tmin,tmax,True)
plot_foof_data(power_mainIllusion,freq_range,tmin,tmax)


#=======exemple avec MNE et topographie : https://fooof-tools.github.io/fooof/auto_examples/analyses/plot_mne_example.html
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

num_sujet = 22
EpochDataMain = load_data_postICA_postdropBad_windows(liste_rawPathMain[num_sujet:num_sujet+1],"",True)
EpochDataPendule = load_data_postICA_postdropBad_windows(liste_rawPathPendule[num_sujet:num_sujet+1],"",True)
EpochDataMainIllusion = load_data_postICA_postdropBad_windows(liste_rawPathMainIllusion[num_sujet:num_sujet+1],"",True)


epoch = EpochDataMainIllusion
montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in epoch:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
from mne.time_frequency import psd_welch
tmin = 1.5
tmax = 10
#dataC3 = EpochDataMain[0].pick_channels(["C3"])
spectra, freqs = psd_welch(epoch[0].average(), fmin=3, fmax=35, tmin=tmin, tmax=tmax,
                           n_overlap=150, n_fft=300)

print(spectra.shape)

# Initialize a FOOOFGroup object, with desired settings
fg = FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
                peak_threshold=2., max_n_peaks=6, verbose=False)

# Define the frequency range to fit
freq_range = [3, 35]
# Fit the power spectrum model across all channels
fg.fit(freqs, spectra, freq_range)
# Check the overall results of the group fits
fg.plot()

# Define frequency bands of interest
bands = Bands({'alpha': [8, 12],
               'SMR': [12,15],
               'beta': [15, 30]})

# Plot the topographies across different frequency bands
import matplotlib.pyplot as plt

def check_nans(data, nan_policy='zero'):
    
    """Check an array for nan values, and replace, based on policy."""

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data

raw = epoch[0].average().drop_channels(["FT9","FT10","TP9","TP10"])
fig, axes = plt.subplots(1,len(bands), figsize=(15, 5))
for ind, (label, band_def) in enumerate(bands):

    # Get the power values across channels for the current band
    band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

    # Create a topomap for the current oscillation band
    mne.viz.plot_topomap(band_power, raw.info, cmap=cm.viridis, contours=0,
                         axes=axes[ind], show=False);

    # Set the plot title
    axes[ind].set_title(label + ' power', {'fontsize' : 20})
    
    
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ind, (label, band_def) in enumerate(bands):

    # Get the power values across channels for the current band
    band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

    # Extracted and plot the power spectrum model with the most band power
    fg.get_fooof(np.argmax(band_power)).plot(ax=axes[ind], add_legend=False)

    # Set some plot aesthetics & plot title
    axes[ind].yaxis.set_ticklabels([])
    axes[ind].set_title('biggest ' + label + ' peak', {'fontsize' : 16})
    
    #plot l'aperiodic
    
# Extract aperiodic exponent values
exps = fg.get_params('aperiodic_params', 'exponent')
# Plot the topography of aperiodic exponents
plot_topomap(exps, raw.info, cmap=cm.viridis, contours=0)
