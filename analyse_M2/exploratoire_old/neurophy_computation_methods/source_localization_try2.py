# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:18:45 2022

@author: claire.dussard
"""

import numpy as np
import mne
from mne.datasets import somato
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd

print(__doc__)

data_path = somato.data_path()
subject = '01'
task = 'somato'
raw_fname = (data_path / f'sub-{subject}' / 'meg' /
             f'sub-{subject}_task-{task}_meg.fif')

# Use a shorter segment of raw just for speed here
raw = mne.io.read_raw_fif(raw_fname)
raw.crop(0, 120)  # one minute for speed (looks similar to using all ~800 sec)

# Read epochs
events = mne.find_events(raw)

epochs = mne.Epochs(raw, events, event_id=1, tmin=-1.5, tmax=2, preload=True)
del raw

# Paths to forward operator and FreeSurfer subject directory
fname_fwd = (data_path / 'derivatives' / f'sub-{subject}' /
             f'sub-{subject}_task-{task}-fwd.fif')


subjects_dir = data_path / 'derivatives' / 'freesurfer' / 'subjects'

#av_power_main

freqs = np.logspace(np.log10(12), np.log10(30), 9)

epochs = EpochDataMain[0]
csd = csd_morlet(epochs, freqs, tmin=-3, tmax=22.6, decim=20)
csd_baseline = csd_morlet(epochs, freqs, tmin=-3, tmax=-1, decim=20)
# ERS activity starts at 0.5 seconds after stimulus onset
csd_ers = csd_morlet(epochs, freqs, tmin=1.5, tmax=22.6, decim=5)
info = epochs.info
del epochs


csd = csd.mean()
csd_baseline = csd_baseline.mean()
csd_ers = csd_ers.mean()

fwd = mne.read_forward_solution(fname_fwd)
filters = make_dics(info, fwd, csd, noise_csd=csd_baseline,
                    pick_ori='max-power', reduce_rank=True, real_filter=True)
del fwd


baseline_source_power, freqs = apply_dics_csd(csd_baseline, filters)
beta_source_power, freqs = apply_dics_csd(csd_ers, filters)


stc = beta_source_power / baseline_source_power
message = 'DICS source power in the 12-30 Hz frequency band'
brain = stc.plot(hemi='both', views='axial', subjects_dir=subjects_dir,
                 subject=subject, time_label=message)

from mne.minimum_norm import make_inverse_operator, apply_inverse
cov = mne.cov.make_ad_hoc_cov(epochs.info)
inv = make_inverse_operator(epochs.info, fwd, cov)

# Apply the inverse model to the trial that also contains the signal.
s = apply_inverse(epochs['signal'].average(), inv)