# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:52:27 2022

@author: claire.dussard
"""

# Import burst detection functions
from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats

# Import simulation code for creating test data
from neurodsp.sim import sim_combined
from neurodsp.utils import set_random_seed, create_times

# Import utilities for loading and plotting data
from neurodsp.utils.download import load_ndsp_data
from neurodsp.plts.time_series import plot_time_series, plot_bursts

#try on data
EpochDataMain = load_data_postICA_postdropBad_windows(liste_rawPathMain,"",True)
EpochDataPendule = load_data_postICA_postdropBad_windows(liste_rawPathPendule,"",True)
epochDataRest = load_data_postICA_preDropbad_effetFBseul(liste_rawPath_rawRest,"",True,True)


n_sujet = 0
elec = "C3"      


#f_range = (8, 12)# f_range = (15,30)
epochs = EpochDataMain

def compute_stats_bursts(f_range,elec,epochs,plot,n_sujet,tmin,tmax):
    epochs = epochs[n_sujet].crop(tmin=tmin,tmax=tmax)
    n_bursts = []
    dur_burst = []
    for i in range(len(epochs)):
        pos_C3 = epochs.info.ch_names.index(elec)
        if epochs.info.ch_names[pos_C3]==elec:
            ep = epochs[i]._data[:,pos_C3]
            if ep.shape[0]==1:
                sig = np.mean(ep,axis=0)
            else:
                sig = ep 
            # Set sampling rate, and create a times vector for plotting
            print(sig.shape)
            fs = 1000
            times = create_times(len(sig)/fs, fs)
            min_n_cycles = 3
            min_duration = None#0.2,0.3,0.6,0.9
            # Detect bursts using the dual threshold algorithm
            bursting = detect_bursts_dual_threshold(sig, fs, (1, 2), f_range,min_n_cycles,min_duration)
            if plot:
                plot_bursts(times, sig, bursting, labels=['Data', 'Detected Burst'])
            burst_stats = compute_burst_stats(bursting, fs)
            n_bursts.append(burst_stats["n_bursts"])
            dur_burst.append(burst_stats["duration_mean"])
        else:
            print("wrong channel")
    return n_bursts,dur_burst

f_range = (11, 15)
elec = "C3"
ls_styles = ["--",":","solid"]
for j in range(1):
    s = 0
    for i in range(j,j+2):
        n_bursts_m,dur_burst_m = compute_stats_bursts(f_range,elec,EpochDataMain,True,i,1.5,26)
        s += 1
    raw_signal.plot(block=True)

n_sujet = 0
data = epochDataRest[n_sujet]
for i in range(len(data)):
    compute_stats_bursts((11,15),"C3",data,True,n_sujet,tmin=1.5,tmax=100)
raw_signal.plot(block=True)