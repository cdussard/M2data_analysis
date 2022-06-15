# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:03:35 2022

@author: claire.dussard
"""
#avec les fonctions MNE
import mne
import numpy as np
import matplotlib.pyplot as plt
#sans les fonctions MNE

av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0]
av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0]
av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]


data = av_power_main.data
data_meanTps = np.mean(data,axis=2)
data_C3 = data_meanTps[11][:]
data_C4 = data_meanTps[13][:]

freqs = np.arange(3, 85, 1)

plt.plot(freqs,data_C3, label='C3') #apres baseline
plt.plot(freqs,data_C4,label="C4")
plt.legend(loc="upper left")
raw_signal.plot(block=True)


#3 conds plot
def plot_elec_cond(av_power,elec_name,cond,elec_position,freqs,fig,ax,scaleMin,scaleMax):
    data = av_power.data
    data_meanTps = np.mean(data,axis=2)
    data_elec = data_meanTps[elec_position][:]
    
    ax.plot(freqs,data_elec, label=elec_name+" "+cond) #apres baseline
    plt.legend(loc="upper left")
    ax.axvline(x=8,color="black",linestyle="--")
    ax.axvline(x=30,color="black",linestyle="--")
    plt.ylim([scaleMin, scaleMax])
    plt.xlim([2, 55])

def plot_3conds(elec_name,elec_pos,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax):
    fig, ax = plt.subplots()
    plot_elec_cond(av_power_pendule,elec_name,"pendule",elec_pos,freqs,fig,ax,scaleMin,scaleMax)
    plot_elec_cond(av_power_main,elec_name,"main",elec_pos,freqs,fig,ax,scaleMin,scaleMax)
    plot_elec_cond(av_power_mainIllusion,elec_name,"mainIllusion",elec_pos,freqs,fig,ax,scaleMin,scaleMax)
    plt.ylim([scaleMin, scaleMax])
    plt.xlim([2, 55])
    
scaleMin = -0.3
scaleMax = 0.05
#left motor cortex
plot_3conds("C3",11,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
plot_3conds("CP5",15,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
plot_3conds("CP1",16,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
plot_3conds("FC1",7,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
plot_3conds("FC5",6,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
raw_signal.plot(block=True)
#les sortir rest VS NFB par sujet

scaleMin = -0.3
scaleMax = 0.05
#right motor cortex (group by electrode)
plot_3conds("C4",13,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
plot_3conds("CP2",17,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
plot_3conds("CP6",18,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
plot_3conds("FC2",8,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
plot_3conds("FC6",9,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
raw_signal.plot(block=True)

#group by condition
def plot_allElec(av_power,condition,elec_names,elec_poses,scaleMin,scaleMax,freqs):
    fig, ax = plt.subplots()
    for elec,pos in zip(elec_names,elec_poses):
        plot_elec_cond(av_power,elec,condition,pos,freqs,fig,ax,scaleMin,scaleMax)
    plt.ylim([scaleMin, scaleMax])
    plt.xlim([2, 55])
        

left_MC = ["C3","CP5","CP1","FC1","FC5"]
left_MC_pos = [11,15,16,7,6]
plot_allElec(av_power_pendule,"pendule",left_MC,left_MC_pos,scaleMin,scaleMax,freqs)
plot_allElec(av_power_main,"main",left_MC,left_MC_pos,scaleMin,scaleMax,freqs)
plot_allElec(av_power_mainIllusion,"mainIllusion",left_MC,left_MC_pos,scaleMin,scaleMax,freqs)

raw_signal.plot(block=True)

scaleMin = -0.35
plot_allElec(av_power_pendule,"pendule",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
plot_allElec(av_power_main,"main",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
plot_allElec(av_power_mainIllusion,"mainIllusion",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)

#for all subjects 

for sujet in range(5):
    num_sujet = sujet
    tfr_pendule_sujet = load_tfr_data_windows(rawPath_pendule_sujets[num_sujet:num_sujet+1],"",True)[0]
    tfr_main_sujet = load_tfr_data_windows(rawPath_main_sujets[num_sujet:num_sujet+1],"",True)[0]
    tfr_mainIllusion_sujet = load_tfr_data_windows(rawPath_mainIllusion_sujets[num_sujet:num_sujet+1],"",True)[0]
    
    
    tfr_pendule_sujet.apply_baseline(mode="logratio",baseline=(-3,-1))
    tfr_main_sujet.apply_baseline(mode="logratio",baseline=(-3,-1))
    tfr_mainIllusion_sujet.apply_baseline(mode="logratio",baseline=(-3,-1))
    
    scaleMin = -0.5
    plot_allElec(tfr_pendule_sujet,"pendule",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
    plot_allElec(tfr_main_sujet,"main",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
    plot_allElec(tfr_mainIllusion_sujet,"mainIllusion",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
raw_signal.plot(block=True)