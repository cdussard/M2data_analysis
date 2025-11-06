# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:53:57 2025

@author: claire.dussard
"""


av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0]
av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0]
av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]

my_cmap = discrete_cmap(13, 'RdBu')
my_cmap_rev = my_cmap.reversed()
tmin = 2.5
tmax = 26.8
v = 0.26
av_power_pendule.plot_topomap(fmin=8,fmax=30,tmin=tmin,tmax=tmax,vmin=-v,vmax=v,cmap=my_cmap_rev,colorbar=True)
av_power_main.plot_topomap(fmin=8,fmax=30,tmin=tmin,tmax=tmax,vmin=-v,vmax=v,cmap=my_cmap_rev,colorbar=True)
av_power_mainIllusion.plot_topomap(fmin=8,fmax=30,tmin=tmin,tmax=tmax,vmin=-v,vmax=v,cmap=my_cmap_rev,colorbar=True)



v = 0.26
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
av_power_pendule.plot_topomap(fmin=8,fmax=30,tmin=tmin,tmax=tmax,vmin=-v,vmax=v,cmap=my_cmap_rev,colorbar=False,axes=axes[0])
av_power_main.plot_topomap(fmin=8,fmax=30,tmin=tmin,tmax=tmax,vmin=-v,vmax=v,cmap=my_cmap_rev,colorbar=False,axes=axes[1])
av_power_mainIllusion.plot_topomap(fmin=8,fmax=30,tmin=tmin,tmax=tmax,vmin=-v,vmax=v,cmap=my_cmap_rev,colorbar=True,axes=axes[2])

plt.tight_layout()
plt.show()


grand_average_MI = mne.time_frequency.read_tfrs("grand_average_MI-tfr.h5")[0]
grand_average_NF = mne.time_frequency.read_tfrs("grand_average_NF-tfr.h5")[0]

v = 0.16
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
grand_average_MI.plot_topomap(fmin=13,fmax=30,tmin=tmin,tmax=tmax,vmin=-v,vmax=v,cmap=my_cmap_rev,colorbar=False,axes=axes[0])
grand_average_NF.plot_topomap(fmin=13,fmax=30,tmin=tmin,tmax=tmax,vmin=-v,vmax=v,cmap=my_cmap_rev,colorbar=True,axes=axes[1])

plt.tight_layout()
plt.show()


v = 0.2
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
grand_average_MI.plot_topomap(fmin=8,fmax=12,tmin=tmin,tmax=tmax,vmin=-v,vmax=v,cmap=my_cmap_rev,colorbar=False,axes=axes[0])
grand_average_NF.plot_topomap(fmin=8,fmax=12,tmin=tmin,tmax=tmax,vmin=-v,vmax=v,cmap=my_cmap_rev,colorbar=True,axes=axes[1])

plt.tight_layout()
plt.show()
