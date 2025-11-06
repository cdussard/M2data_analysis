# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:03:08 2023

@author: claire.dussard
"""

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
epochs = load_data_postICA_postdropBad_windows(liste_rawPathMain[0:1],"",True)[0]
#set montage
montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
epochs.set_eeg_reference(projection=True)  # needed for inverse modeling
epochs.set_montage(montageEasyCap)
epochs.crop(tmin=2,tmax=24)

#compute noise covar
noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, epochs.info)

#compute evoked (nous c'est pas ca)
evoked = epochs.average().pick('eeg')
evoked.plot(time_unit='s')
evoked.plot_topomap(times=np.linspace(2.5, 25, 5), ch_type='eeg')

#checknoise
evoked.plot_white(noise_cov, time_unit='s')


# #load fs average MRI 
# fs_dir = fetch_fsaverage(verbose=True)
# subjects_dir = op.dirname(fs_dir)

# # The files live in:
# subject = 'fsaverage'
# trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
# src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
# bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


# # Check that the locations of EEG electrodes is correct with respect to MRI
# mne.viz.plot_alignment(
#     epochs.info, src=src, eeg=['original', 'projected'], trans=trans,
#     show_axes=True, mri_fiducials=True, dig='fiducials')

#COMPUTE forward solution
fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=None)
print(fwd)

#compute inverse operator
inverse_operator = make_inverse_operator(
    evoked.info, fwd, noise_cov, loose=0.2, depth=0.8)
del fwd

method = "dSPM"
#method = "sLORETA"
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)

#plot the solution
fig, ax = plt.subplots()
ax.plot(1e3 * stc.times, stc.data[::100, :].T)
ax.set(xlabel='time (ms)', ylabel='%s value' % method)

vertno_max, time_max = stc.get_peak(hemi='rh')

subjects_dir = data_path / 'subjects'
surfer_kwargs = dict(
    hemi='rh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, method+' (plus location of maximal activation)', 'title',
               font_size=14)