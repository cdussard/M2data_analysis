# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:18:56 2022

@author: claire.dussard
"""

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
raw_fnames = eegbci.load_data(1, 1)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raws)
mne.datasets.eegbci.standardize(raw)  # set channel names
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

liste_power_sujets = []
freqs = np.arange(3,85 , 1)

events, _ = mne.events_from_annotations(raw)
ep_raw = mne.Epochs(raw,events)
raw_tfr = mne.time_frequency.tfr_morlet(ep_raw,freqs=freqs,n_cycles=1,return_itc=False)

anim = animate_topo(raw_tfr,False,0,0.4,0.01)

av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]

def animate_topo(data,doBaseline,tmin,tmax,step):
    if doBaseline:
        data.apply_baseline(baseline=(-3,-1),mode="logratio")
    data_copy = data.copy()
    data_copy.crop(fmin=8,fmax=30)
    
    data_8_30 = np.mean(data_copy.data,axis=1)
    
    epo = EvokedArray(data_8_30, data.info)
    
    times = np.arange(tmin,tmax,step)
    fig,anim = epo.animate_topomap(
        times=times, ch_type='eeg', frame_rate=3, time_unit='s', blit=False,show=False)
    raw_signal.plot(block=True)
    return anim

anim = animate_topo(av_power_pendule,False,6,24,1)
anim.save('animation_pendule.gif', writer='imagemagick', fps=3)#s011 main