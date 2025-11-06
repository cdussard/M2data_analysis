# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 13:18:02 2023

@author: claire.dussard
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 19:40:33 2022

@author: claire.dussard
"""

import os 
import seaborn as sns
import pathlib
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *
import numpy as np 
from plotPsd import * 

#create liste of file paths
essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData() 

nom_essai = "5-1"
#on prend deuxieme resting state
nom_essai = "1-2"
noms_essais = ["2-1-2","2-b"]+["2-2" for i in range(21)]
#on commence au 3 (reste pbs noms)
liste_rawPath_thresh = []
#i = 0
for num_sujet in allSujetsDispo:
    print("sujet n° "+str(num_sujet))
    nom_sujet = listeNumSujetsFinale[num_sujet]
    if num_sujet in SujetsPbNomFichiers:
        if num_sujet>0:
            date_sujet = '19-04-2021'
        else:
            date_sujet = '15-04-2021'
    else:
        date_sujet = dates[num_sujet]
    date_nom_fichier = date_sujet[-4:]+"-"+date_sujet[3:5]+"-"+date_sujet[0:2]+"_"
    dateSession = listeDatesFinale[num_sujet]
    sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
    sample_data_dir = pathlib.Path(sample_data_loc)
    raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + nom_essai+".vhdr")#il faudrait recup les epochs et les grouper ?
    liste_rawPath_thresh.append(raw_path_sample)
    #i += 1

print(liste_rawPath_thresh)

os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

sujetsNonDispo = [10,17,24]
liste_rawPath_thresh.pop(22)#24
liste_rawPath_thresh.pop(15)#17
liste_rawPath_thresh.pop(8)#10
#compute values in 13-20Hz window and 20-30Hz window
#epochs the signal
liste_data_baseline= read_raw_data(liste_rawPath_thresh)


i = 0
for signal in liste_data_baseline:
    print(mne.events_from_annotations(signal))
    if 12 not in mne.events_from_annotations(signal)[0][:,2]:
        liste_data_baseline.pop(i)
        print("popped")
    print(i)
    i += 1
    

def report_medians(lowFreq,highFreq):
    event_id={'Begin threshold rest':12}
    liste_data = liste_data_baseline
    epochs = epoching(event_id,liste_data,liste_data,120,0)
    #compute laplacian
    liste_laplacian_epochs = []
    for epoch in epochs:
        laplacien = compute_laplacian(epoch)
        liste_laplacian_epochs.append(laplacien)
    liste_laplacian_epochs = liste_laplacian_epochs[0]
    #filter it 
    for epoch in liste_laplacian_epochs:
        epoch.filter(lowFreq,highFreq)
        epoch.crop(tmin=1)
    #square signal
    for epoch in liste_laplacian_epochs:
        epoch._data = epoch._data*epoch._data
        epoch.pick_channels(["C3"])
    #average over time windows (computeMovingAverage)
    liste_medians = []
    for epoch in liste_laplacian_epochs:
        print(epoch._data.shape)
        test = np.mean(epoch._data,0)
        print(test.shape)
        test = np.mean(test,0)
        print(test.shape)
        points = computeMovingAverage(test,90)
        print(points)
        median = np.median(points[1:])
        print(median)
        liste_medians.append(median)
    return liste_medians

ls_lowBeta = report_medians(13,20)
ls_highBeta=report_medians(21,30)
#compute ratio
lowOverHigh = [a/b for a,b in zip(ls_lowBeta,ls_highBeta) ]
#positif : low > high

result_30sCalib = [1.0896, 0.6797, 1.28545, 0.72385, 1.1703,
 0.716459, 1.344616, 0.84002, 0.125052, 0.94609,
 1.64110, 1.03503, 1.17779, 1.39862, 1.40792,
 0.85426, 1.40036, 1.66733, 1.46332, 1.70337]
result_30sCalib.pop(13)


#do the same with the 2 min rest recording
result_2minRestingState = [1.0575, 0.8622, 1.6547, 0.9022,
 1.366, 1.1166, 1.1244, 1.0488,
 1.1815, 0.968, 1.1869, 1.5457,
 1.2104, 1.5676, 0.9605, 1.1662,
 0.9099, 1.3513, 0.692, 1.4892, 1.1249, 0.6757]
result_2minRestingState.pop(21)#24
result_2minRestingState.pop(14)#17
result_2minRestingState.pop(8)#10

plt.plot(result_30sCalib,result_2minRestingState)

plt.scatter(result_30sCalib,result_2minRestingState)
plt.axline((0, 0), slope=1)
signal.plot(block=True)


#using YASA
#do the same after fitting the 1/f curve
import yasa
#a appliquer sur les raw data
ch = "C3"
laplacians = ['CP5','CP1','FC5', 'FC1']
ch = "C4"
laplacians = ['CP6','CP2','FC6', 'FC2']
for data_suj in liste_data_baseline:
    raw = data_suj.pick_types(eeg=True)
    
    # Extract data, sf, and chan
    data = raw.get_data(units="uV")
    sf = raw.info['sfreq']
    chan = raw.ch_names
    
    from scipy.signal import welch
    
    win = int(4 * sf)  # Window size is set to 4 seconds
    freqs, psd = welch(data, sf, nperseg=win)  # Works with single or multi-channel data
    plot_power(freqs,psd,ch,1,40)
    
    #with the laplacian
    new_epoch_C3 = 4*raw.copy().pick_channels([ch]).get_data()
    
    for i in range(len(laplacians)):
        new_epoch_C3 = new_epoch_C3-raw.copy().pick_channels([laplacians[i]]).get_data()
    
    freqs2, psd2 = welch(new_epoch_C3, sf, nperseg=win)  # Works with single or multi-channel data
    plt.plot(freqs2, psd2[0, :], 'k', lw=2)
    plt.fill_between(freqs2, psd2[0,:], cmap='Spectral')
    plt.xlim(fmin, fmax)
    plt.yscale('log')
    sns.despine()
    plt.title("Laplacian"+ch)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)')
    raw.plot(block=True)

# Plot
ch = "C3"

def plot_power(freqs,psd,ch,fmin,fmax):
    num = chan.index(ch)
    print(num)
    plt.plot(freqs, psd[num, :], 'k', lw=2)
    plt.fill_between(freqs, psd[num, :], cmap='Spectral')
    plt.xlim(fmin, fmax)
    plt.yscale('log')
    sns.despine()
    plt.title(chan[num])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)')

plot_power(freqs,psd,"C3",1,40)
plot_power(freqs,psd,"Cz",1,40)
plot_power(freqs,psd,"C4",1,40)
raw.plot(block=True)

plot_power(freqs,psd,"FC1",1,40)
plot_power(freqs,psd,"CP5",1,40)
plot_power(freqs,psd,"C3",1,40)
plot_power(freqs,psd,"CP1",1,40)
plot_power(freqs,psd,"FC5",1,40)
raw.plot(block=True)


# new_epoch_C3 = 4*raw.copy().pick_channels(["C3"]).get_data()
# laplacians = ['CP5','CP1','FC5', 'FC1']
# for i in range(len(laplacians)):
#     new_epoch_C3 = new_epoch_C3-raw.copy().pick_channels([laplacians[i]]).get_data()


# freqs2, psd2 = welch(new_epoch_C3, sf, nperseg=win)  # Works with single or multi-channel data
# plt.plot(freqs2, psd2[0, :], 'k', lw=2)
# plt.fill_between(freqs2, psd2[0,:], cmap='Spectral')
# plt.xlim(fmin, fmax)
# plt.yscale('log')
# sns.despine()
# plt.title("Laplacian C3")
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD log($uV^2$/Hz)')
# raw.plot(block=True)



#++++++++++++++++++++++++++


#with MNE fct
from mne.viz import iter_topography
tmin, tmax = 0, 120  # use the first 120s of data
fmin, fmax = 2, 40  # look at frequencies between 2 and 20Hz

raw = liste_data_baseline[0].pick_types(eeg=True)
raw.drop_channels(["ECG","EMG"])
montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
raw.set_montage(montageEasyCap)

spectrum = raw.compute_psd(
     tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
psds, freqs = spectrum.get_data(exclude=(), return_freqs=True)
psds = 20 * np.log10(psds)  # scale to dB

def my_callback(ax, ch_idx):
    """
    This block of code is executed once you click on one of the channel axes
    in the plot. To work with the viz internals, this function should only take
    two parameters, the axis and the channel or data index.
    """
    ax.plot(freqs[20:180], psd[ch_idx][20:180], color='red')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')


for ax, idx in iter_topography(raw.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback):
    ax.plot(psd[idx], color='red')

plt.gcf().suptitle('Power spectral densities')
plt.show()


