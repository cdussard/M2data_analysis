# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:15:34 2023

@author: claire.dussard
"""
#try their code
import numpy as np
import mne
from mne.cov import compute_covariance
from mne.time_frequency import csd_morlet
import os.path as op
from mne.datasets import fetch_fsaverage
from mne.beamformer import make_dics
#LAST TRY FOR FORWARD WITH EEG ELECTRODES
raw = test
fs_dir = fetch_fsaverage(verbose=True)
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=None)
print(fwd)

#check alignment
mne.viz.plot_alignment(
    raw.info, src=src, eeg=['original', 'projected'], trans=trans,
    show_axes=True, mri_fiducials=True, dig='fiducials')

stcs = []

#params
fmin = 12.5
fmax = 30
tmin_bl = -5
tmin_erd = 5
tmax = 25
for i in range(len(liste_rawPathMain)):
    num_sujet = i
    liste_cond = liste_rawPathMainIllusion #liste_rawPathMain #liste_rawPathMainIllusion
    
    
    # #get the forward operator
    
    # epochDataMain_dropBad = load_data_postICA_preDropbad(liste_cond[i:i+1],"",True)[0]
    # epochs = epochDataMain_dropB ad.pick_types(eeg=True)
    # epochs_ = mne.add_reference_channels(epochDataMain_dropBad,"Fz")
    # epochs=epochs_.set_eeg_reference("average") # needed for inverse modeling
    
    epochs = load_data_postICA_postdropBad_windows(liste_cond[i:i+1],"",True)[0]
    
    montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
    epochs.set_montage(montageEasyCap)
    
    
    # signalInitialRef_main = mne.add_reference_channels(epochDataMain_dropBad,"Fz")
    # averageRefSignal_main = signalInitialRef_main.set_eeg_reference('average')
    
    
    epochs.filter(fmin,fmax)
    epochs.crop(tmin=tmin_bl,tmax=tmax)
    
    
    rank = mne.compute_rank(epochs, tol=1e-6, tol_kind='relative')
    
    
    
    # Compute source estimates
    baseline_win = (-3, -1)
    active_win = (tmin_erd,tmax)
    stc_dics = _gen_dics(active_win, baseline_win, epochs)
    
    stcs.append(stc_dics)

brain_dics = stc_dics.plot(
    hemi='both', subjects_dir=None, subject=None,
    time_label='DICS source power in the'+str(fmin)+'-'+str(fmax)+' Hz frequency band')

#stc_fsaverage = stc_dics.morph('fsaverage')
#stcs = [stc_dics,stc_dics]

data = np.mean([stc.data for stc in stcs], axis=0)
ga_stc = mne.SourceEstimate(data, vertices=stcs[0].vertices,
                                tmin=stcs[0].tmin, tstep=stcs[0].tstep)

brain_dics = ga_stc.plot(
    hemi='both', subjects_dir=None, subject="fsaverage",
    time_label='DICS source power in the'+str(fmin)+'-'+str(fmax)+' Hz frequency band')

def _gen_dics(active_win, baseline_win, epochs):
    freqs = np.arange(fmin, fmax, 1)
    #freqs = np.logspace(np.log10(fmin), np.log10(fmax), 9)
    csd = csd_morlet(epochs, freqs, tmin=baseline_win[0], tmax=tmax, decim=20,n_cycles=freqs)#ajout n_cycles = freqs
    csd_baseline = csd_morlet(epochs, freqs, tmin=baseline_win[0],
                              tmax=baseline_win[1], decim=20,n_cycles=freqs)
    csd_erd = csd_morlet(epochs, freqs, tmin=active_win[0], tmax=active_win[1],
                         decim=20,n_cycles=freqs)
    filters = make_dics(epochs.info, fwd, csd.mean(), pick_ori='max-power',
                        reduce_rank=True, real_filter=True, rank=rank,weight_norm='unit-noise-gain-invariant')
    #avant de faire la moyenne, normaliser l'orientation des dipoles ?
    #volume normalization : "unit-noise-gain-invariante ??
    stc_base, freqs = apply_dics_csd(csd_baseline.mean(), filters)
    stc_act, freqs = apply_dics_csd(csd_erd.mean(), filters)
    stc_act /= stc_base
    return stc_act