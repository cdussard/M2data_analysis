# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:09:22 2022

@author: claire.dussard
"""
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from matplotlib.colors import TwoSlopeNorm
from mne.time_frequency import tfr_multitaper
freqs = np.arange(3, 40)  # frequencies from 2-35Hz
vmin, vmax = -0.1,0.1 # set min and max ERDS values in plot
baseline = [-3, -1]  # baseline interval (in s)
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None, out_type='mask')  # for cluster test

epochs = load_data_postICA_postdropBad_windows(liste_rawPathMain[0:1],"",True)

epochs[0].pick_channels(["C3","Cz","C4"])
tfr = tfr_multitaper(epochs[0], freqs=freqs, n_cycles=freqs, use_fft=True,
                     return_itc=False, average=False, decim=2)
tfr.crop(-4, 28).apply_baseline(baseline, mode="logratio")

# tfr = av_power_main
# tfr =  av_power_main.copy().pick_channels(["C3"])


# select desired epochs for visualization
tfr_ev = tfr
fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 1]})
ch = 0
ax = axes[0]
for ch, ax in enumerate(axes[:-1]):  # for each channel
# positive clusters
    _, c1, p1, _ = pcluster_test(tfr_ev.data[ch,:], tail=1, **kwargs)
    # negative clusters
    _, c2, p2, _ = pcluster_test(tfr_ev.data[ch,:], tail=-1, **kwargs)
    
    # note that we keep clusters with p <= 0.05 from the combined clusters
    # of two independent tests; in this example, we do not correct for
    # these two comparisons
    c = np.stack(c1 + c2, axis=2)
  # combined clusters
    p = np.concatenate((p1, p2))  # combined p-values
    mask = c[..., p <= 0.05].any(axis=-1)
    
    # plot TFR (ERDS map with masking)
    tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
                          colorbar=False, show=False, mask=mask,
                          mask_style="mask")
    
    ax.set_title(epochs[0].ch_names[ch], fontsize=10)
    ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
    if ch != 0:
        ax.set_ylabel("")
        ax.set_yticklabels("")
fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")

plt.show()
raw_signal.plot(block=True)
