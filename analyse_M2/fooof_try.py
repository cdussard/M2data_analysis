# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:20:51 2022

@author: claire.dussard
"""

import fooof
import mne
import numpy as np
from fooof import FOOOF
from functions.load_savedData import *
import pandas as pd

# Initialize a FOOOF object
fm = FOOOF()

#load data
# Import the FOOOF object
from fooof import FOOOF
from fooof import FOOOFGroup



def plot_foof_data(power_object,freq_range,tmin,tmax,keepObject,aperiodic_mode):
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

    fm = FOOOF(peak_width_limits=(mean_peak,12),aperiodic_mode=aperiodic_mode)
    fm.fit(freqs, test, freq_range)
    fm.report(freqs, test, freq_range)
    return fm
    #fm.print_results()
    #fm.plot()
    
def plot_foofGroup_data(power_object,freq_range,tmin,tmax,keepObject):
    if keepObject:
        power_object = power_object.copy()
    power_object.pick_channels(["C3"])
    power_object.crop(tmin=tmin,tmax=tmax,fmin=min(freq_range),fmax=max(freq_range))
    freqs = power_object.freqs
    if min(freqs)==min(freq_range) and max(freqs)==max(freq_range):
        spectra = np.mean(power_object.data,axis=3)#mean over time
        spectra = np.mean(spectra,axis=1)#mean over C3 electrode
        fg = FOOOFGroup(peak_width_limits=[2, 12], min_peak_height=0.1, max_n_peaks=4)
        fg.fit(freqs, spectra, freq_range)
        fg.report(freqs, spectra, freq_range)
        return fg
    else:
        print("ISSUE : input freq range not contained in original data")


        
num_sujet = 22 #entre 0 et 22
power_pendule = load_tfr_data_windows(liste_rawPathPendule[num_sujet:num_sujet+1],"allTrials",True)[0]
power_main = load_tfr_data_windows(liste_rawPathMain[num_sujet:num_sujet+1],"allTrials",True)[0]
power_mainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion[num_sujet:num_sujet+1],"allTrials",True)[0]


freq_range = [3, 45]
tmin = 5
tmax = 30
i = 0
for i in range(len(power_pendule)):
    plot_foof_data(power_pendule[i].average(),freq_range,tmin,tmax,True)

plot_foof_data(power_main.average(),freq_range,tmin,tmax,True)
plot_foof_data(power_mainIllusion.average(),freq_range,tmin,tmax,True)


plot_foof_data(power_pendule.average(),freq_range,-5,0,True)
plot_foof_data(power_pendule.average(),freq_range,5,25,True,"fixed")

plot_foof_data(power_main.average(),freq_range,tmin,tmax,True)
plot_foof_data(power_mainIllusion.average(),freq_range,tmin,tmax,True)
#EpochDataMain[0].plot(block=True)

#==================pour un sujet ============================
    
sujets_epochs_jetes_main = [
    [],[],[],[3],[2,4,5,6,7],[7],[],[6],[9,10],[8],[6],
    [1,6,8],[1,10],[9,10],[6,7,8,9,10],[3,6],[3,6,7],[4,10],[],[1,6],[],[9],[]
    ]

sujets_epochs_jetes_pendule = [
    [],[],[],[5],[1,7,10],[],[],[3,5,8,10],[],[5,10],[],
    [5,6],[4],[6,9],[],[9],[3,8,9],[],[],[1,6],[6],[3,9],[6,8]
    ]

sujets_epochs_jetes_mainIllusion = [
    [6],[1,3,6],[1,2],[],[5,6,8,9,10],[],[],[1,6,7,8],[6,7,8,9,10],[4,10],[1],
    [],[1,8,10],[10],[6,9],[9],[4,8,9],[4,8],[],[1,6],[],[1],[]
    ]

def return_range_dispo_cond(epochs_jetes_cond):
    range_dispo = [i for i in range(1,11)]
    if len(epochs_jetes_cond)>0: 
        for item in epochs_jetes_cond:
            range_dispo.remove(item)
    return range_dispo

dispo_main = []
for i in range(23):
    dispo_main.append(return_range_dispo_cond(sujets_epochs_jetes_main[i]))
    
dispo_pendule = []
for i in range(23):
    dispo_pendule.append(return_range_dispo_cond(sujets_epochs_jetes_pendule[i]))

dispo_mainIllusion = []
for i in range(23):
    dispo_mainIllusion.append(return_range_dispo_cond(sujets_epochs_jetes_mainIllusion[i]))

def return_sujet_data_cond(num_sujet,name_cond,power_cond_allTrials,freq_range,tmin,tmax,essais_dispo):
    df_cond = pd.DataFrame(columns=["num_sujet","num_essai","FB","freq","hauteur","largeur"])
    fg = plot_foofGroup_data(power_cond_allTrials,freq_range,tmin,tmax,True)
    for i in range(len(power_cond_allTrials)):
        fm = fg.get_fooof(ind=i, regenerate=True)
        params_peak = fm.get_params("peak_params")
        for ligne in params_peak:
            data_peak = {
                "num_sujet":num_sujet,
                "num_essai":essais_dispo[i],#modifie
                "FB":name_cond,
                "freq":round(ligne[0],1),
                "hauteur":round(ligne[1],2),
                "largeur":round(ligne[2],2),
                "Rsquared_fit":round(fm.get_params("r_squared"),2),
                "aperiodic_exp":round(fm.get_params("aperiodic_params")[1],2),
                "aperiodic_offset":round(fm.get_params("aperiodic_params")[0],2)
                }
            print(data_peak)
            df_cond = df_cond.append(data_peak,ignore_index=True)
    return df_cond
       
def return_sujet_data(num_sujet,freq_range,tmin,tmax):
    #read data
    print("loading data")
    power_pendule = load_tfr_data_windows(liste_rawPathPendule[num_sujet:num_sujet+1],"allTrials",True)[0]
    power_main = load_tfr_data_windows(liste_rawPathMain[num_sujet:num_sujet+1],"allTrials",True)[0]
    power_mainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion[num_sujet:num_sujet+1],"allTrials",True)[0]
    #create dataframe
    df_3cond = pd.DataFrame(columns=["num_sujet","num_essai","FB","freq","hauteur","largeur"])
    #fit FOOOF, return peak params
    df_pendule_sujet = return_sujet_data_cond(num_sujet,"pendule",power_pendule,freq_range,tmin,tmax,dispo_pendule[num_sujet])  
    df_main_sujet = return_sujet_data_cond(num_sujet,"main",power_main,freq_range,tmin,tmax,dispo_main[num_sujet])   
    df_mainIllusion_sujet = return_sujet_data_cond(num_sujet,"mainIllusion",power_mainIllusion,freq_range,tmin,tmax,dispo_mainIllusion[num_sujet])  
    df_3cond = pd.concat([df_pendule_sujet, df_main_sujet, df_mainIllusion_sujet], ignore_index=True)
    return df_3cond

df_full = pd.DataFrame(columns=["num_sujet","num_essai","FB","freq","hauteur","largeur"])
for i in range(16,23):
    df_3cond = return_sujet_data(i,freq_range,5,25)
    df_full = pd.concat([df_full,df_3cond])
print(df_full)

df_full.to_csv("../csv_files/FOOOF_analysis_byTrial_essaisFixed.csv")
        


#pendule
fm_p = plot_foof_data(power_pendule.average(),freq_range,5,25,True,"fixed")
params_peak_p = fm_p.get_params("peak_params")#en ligne les peaks avec freq, pw,bw
fg_p = plot_foofGroup_data(power_pendule,freq_range,5,25,True)
fg_p.plot()
for i in range(10):
    fm = fg_p.get_fooof(ind=i, regenerate=True)
    fm.print_results()
    fm.plot()


#main
fm_m = plot_foof_data(power_main.average(),freq_range,5,25,True,"fixed")
params_peak_m = fm_m.get_params("peak_params")#en ligne les peaks avec freq, pw,bw
fg_m = plot_foofGroup_data(power_main,freq_range,5,25,True)
fg_m.plot()
for i in range(10):
    fm = fg_m.get_fooof(ind=i, regenerate=True)
    fm.print_results()
    fm.plot()


#mainIllusion
fm_mi = plot_foof_data(power_mainIllusion.average(),freq_range,5,25,True,"fixed")
params_peak_mi = fm_mi.get_params("peak_params")#en ligne les peaks avec freq, pw,bw
fg_mi = plot_foofGroup_data(power_mainIllusion,freq_range,5,25,True)
fg_mi.plot()
for i in range(10):
    fm = fg_mi.get_fooof(ind=i, regenerate=True)
    fm.print_results()
    fm.plot()

#get values for 8-30Hz band



# #=======exemple avec MNE et topographie : https://fooof-tools.github.io/fooof/auto_examples/analyses/plot_mne_example.html
# from fooof import FOOOFGroup
# from fooof.bands import Bands
# from fooof.analysis import get_band_peak_fg
# from fooof.plts.spectra import plot_spectrum

# num_sujet = 0
# EpochDataMain = load_data_postICA_postdropBad_windows(liste_rawPathMain[num_sujet:num_sujet+1],"",True)
# EpochDataPendule = load_data_postICA_postdropBad_windows(liste_rawPathPendule[num_sujet:num_sujet+1],"",True)
# EpochDataMainIllusion = load_data_postICA_postdropBad_windows(liste_rawPathMainIllusion[num_sujet:num_sujet+1],"",True)


# #yo = fm.fit(power_pendule.freqs, EpochDataMain[0][0].average()._data[13], [3,84])

# epoch = EpochDataMain
# montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
# for epochs in epoch:
#     if epochs!=None:
#         epochs.set_montage(montageEasyCap)
# from mne.time_frequency import psd_welch
# tmin = -3
# tmax = -1
# #dataC3 = EpochDataMain[0].pick_channels(["C3"])
# spectra, freqs = psd_welch(epoch[0].average(), fmin=3, fmax=35, tmin=tmin, tmax=tmax,
#                            n_overlap=150, n_fft=300)

# print(spectra.shape)

# # Initialize a FOOOFGroup object, with desired settings
# fg = FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
#                 peak_threshold=2., max_n_peaks=6, verbose=False)

# # Define the frequency range to fit
# freq_range = [3, 35]
# # Fit the power spectrum model across all channels
# fg.fit(freqs, spectra, freq_range)
# # Check the overall results of the group fits
# fg.plot()

# # Define frequency bands of interest
# bands = Bands({'alpha': [8, 12],
#                'SMR': [12,15],
#                'beta': [15, 30]})

# # Plot the topographies across different frequency bands
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# def check_nans(data, nan_policy='zero'):
    
#     """Check an array for nan values, and replace, based on policy."""

#     # Find where there are nan values in the data
#     nan_inds = np.where(np.isnan(data))

#     # Apply desired nan policy to data
#     if nan_policy == 'zero':
#         data[nan_inds] = 0
#     elif nan_policy == 'mean':
#         data[nan_inds] = np.nanmean(data)
#     else:
#         raise ValueError('Nan policy not understood.')

#     return data

# raw = epoch[0].average().drop_channels(["FT9","FT10","TP9","TP10"])
# fig, axes = plt.subplots(1,len(bands), figsize=(15, 5))
# for ind, (label, band_def) in enumerate(bands):

#     # Get the power values across channels for the current band
#     band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

#     # Create a topomap for the current oscillation band
#     mne.viz.plot_topomap(band_power, raw.info, cmap=cm.viridis, contours=0,
#                          axes=axes[ind], show=False);

#     # Set the plot title
#     axes[ind].set_title(label + ' power', {'fontsize' : 20})
    
    
# fig, axes = plt.subplots(1, 3, figsize=(15, 6))
# for ind, (label, band_def) in enumerate(bands):

#     # Get the power values across channels for the current band
#     band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

#     # Extracted and plot the power spectrum model with the most band power
#     fg.get_fooof(np.argmax(band_power)).plot(ax=axes[ind], add_legend=False)

#     # Set some plot aesthetics & plot title
#     axes[ind].yaxis.set_ticklabels([])
#     axes[ind].set_title('biggest ' + label + ' peak', {'fontsize' : 16})
    
#     #plot l'aperiodic
    
# # Extract aperiodic exponent values
# exps = fg.get_params('aperiodic_params', 'exponent')
# # Plot the topography of aperiodic exponents
# mne.viz.plot_topomap(exps, raw.info, cmap=cm.viridis, contours=0)


# liste_power_sujets = load_tfr_data_windows(liste_rawPath_rawRest,"",True)

# for av_power in liste_power_sujets:
#     freqs = np.arange(3, 41, 1)
#     freq_range = [3, 40]
#     av_power_copy = av_power.copy()
#     av_power_copy.crop(tmin=5,tmax=25,fmin = 3,fmax = 40)
#     data = av_power_copy.data
#     data_elec = data[elec_position]
#     data_meanTps = np.mean(data_elec,axis=1)
    
#     # Report: fit the model, print the resulting parameters, and plot the reconstruction
#     #fm.report(freqs, data_meanTps, freq_range)
    
#     # Import the FOOOF object
#     fm.fit(freqs, data_meanTps, freq_range)
    
#     # After fitting, plotting and parameter fitting can be called independently:
#     fm.print_results()
#     fm.plot()
# raw_signal.plot(block=True)