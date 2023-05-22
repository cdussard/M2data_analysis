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

#3 conds plot
def plot_elec_cond(av_power,elec_name,cond,elec_position,freqs,fig,ax,scaleMin,scaleMax,tmin,tmax,fmin,fmax,color,label):
    ch_names = av_power.info.ch_names
    if ch_names[elec_position]!=elec_name:
        print(av_power.info.ch_names[elec_position]+"IS NOT "+elec_name)
        elec_position = ch_names.index(elec_name)
    print(ch_names[elec_position]==elec_name)
    av_power_copy = av_power.copy()
    av_power_copy.crop(tmin=tmin,tmax=tmax)
    data = av_power_copy.data
    print(data.shape)
    data_meanTps = np.mean(data,axis=2)
    data_elec = data_meanTps[elec_position][:]
    if label:
        if color is not None:
            ax.plot(freqs,data_elec, label=elec_name+" "+cond,color=color)
        else:
            ax.plot(freqs,data_elec, label=elec_name+" "+cond)
            
    else:
        ax.plot(freqs,data_elec,color=color) 
    plt.legend(loc="upper left")
    ax.axvline(x=8,color="black",linestyle="--")
    ax.axvline(x=30,color="black",linestyle="--")
    plt.ylim([scaleMin, scaleMax])
    plt.xlim([fmin, fmax])


def plot_3conds(elec_name,elec_pos,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax):
    freqs = np.arange(3, 85, 1)
    fig, ax = plt.subplots()
    plot_elec_cond(av_power_pendule,elec_name,"pendule",elec_pos,freqs,fig,ax,scaleMin,scaleMax,1.5,26.5)
    plot_elec_cond(av_power_main,elec_name,"main",elec_pos,freqs,fig,ax,scaleMin,scaleMax,1.5,26.5)
    plot_elec_cond(av_power_mainIllusion,elec_name,"mainIllusion",elec_pos,freqs,fig,ax,scaleMin,scaleMax,1.5,26.5)
    plt.ylim([scaleMin, scaleMax])
    plt.xlim([2, 55])
    
    #group by condition
def plot_allElec(av_power,condition,elec_names,elec_poses,scaleMin,scaleMax,freqs):
    fig, ax = plt.subplots()
    for elec,pos in zip(elec_names,elec_poses):
        plot_elec_cond(av_power,elec,condition,pos,freqs,fig,ax,scaleMin,scaleMax,1.5,26.5)
    plt.ylim([scaleMin, scaleMax])
    plt.xlim([2, 55])
            

# av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0]
# av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0]
# av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]


# data = av_power_main.data
# data_meanTps = np.mean(data,axis=2)
# data_C3 = data_meanTps[11][:]
# data_C4 = data_meanTps[13][:]

# freqs = np.arange(3, 85, 1)

# plt.plot(freqs,data_C3, label='C3') #apres baseline
# plt.plot(freqs,data_C4,label="C4")
# plt.legend(loc="upper left")
# raw_signal.plot(block=True)

    
# scaleMin = -0.4
# scaleMax = 0.05
# #left motor cortex
# plot_3conds("C3",11,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
# plot_3conds("CP5",15,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
# plot_3conds("CP1",16,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
# plot_3conds("FC1",7,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
# plot_3conds("FC5",6,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
# raw_signal.plot(block=True)
# #les sortir rest VS NFB par sujet

# scaleMin = -0.4
# scaleMax = 0.05
# #right motor cortex (group by electrode)
# plot_3conds("C4",13,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
# plot_3conds("CP2",17,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
# plot_3conds("CP6",18,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
# plot_3conds("FC2",8,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
# plot_3conds("FC6",9,av_power_pendule,av_power_main,av_power_mainIllusion,scaleMin,scaleMax)
# raw_signal.plot(block=True)



# left_MC = ["C3","CP5","CP1","FC1","FC5"]
# left_MC_pos = [11,15,16,7,6]
# plot_allElec(av_power_pendule,"pendule",left_MC,left_MC_pos,scaleMin,scaleMax,freqs)
# plot_allElec(av_power_main,"main",left_MC,left_MC_pos,scaleMin,scaleMax,freqs)
# plot_allElec(av_power_mainIllusion,"mainIllusion",left_MC,left_MC_pos,scaleMin,scaleMax,freqs)

# raw_signal.plot(block=True)

# scaleMin = -0.4
# plot_allElec(av_power_pendule,"pendule",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
# plot_allElec(av_power_main,"main",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
# plot_allElec(av_power_mainIllusion,"mainIllusion",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)

# raw_signal.plot(block=True)

# #for all subjects 

# for sujet in range(5):
#     num_sujet = sujet
#     tfr_pendule_sujet = load_tfr_data_windows(rawPath_pendule_sujets[num_sujet:num_sujet+1],"",True)[0]
#     tfr_main_sujet = load_tfr_data_windows(rawPath_main_sujets[num_sujet:num_sujet+1],"",True)[0]
#     tfr_mainIllusion_sujet = load_tfr_data_windows(rawPath_mainIllusion_sujets[num_sujet:num_sujet+1],"",True)[0]
    
    
#     tfr_pendule_sujet.apply_baseline(mode="logratio",baseline=(-3,-1))
#     tfr_main_sujet.apply_baseline(mode="logratio",baseline=(-3,-1))
#     tfr_mainIllusion_sujet.apply_baseline(mode="logratio",baseline=(-3,-1))
    
#     scaleMin = -0.5
#     plot_allElec(tfr_pendule_sujet,"pendule",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
#     plot_allElec(tfr_main_sujet,"main",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
#     plot_allElec(tfr_mainIllusion_sujet,"mainIllusion",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
# raw_signal.plot(block=True)


# #afficher la meme chose mais avec un intervalle de confiance sur les sujets / par sujet
# import numpy as np 
# from scipy.stats import t
# import scipy

def plot_oneCond_freqDesync(nomCond,colorPlot,fig,ax,fmax,confidence,num_elec):
    #load all subjects
    #S00
    liste_C3 = []
    listeNumSujetsFinale_mod = listeNumSujetsFinale[0:1]+listeNumSujetsFinale[2:]
    listeNumSujetsFinale_mod = listeNumSujetsFinale_mod[0:3]+listeNumSujetsFinale_mod[4:]
    data_freq_suj = np.zeros(shape=(23,82))
    for suj in range(23):
        print(suj)
        name_suj = listeNumSujetsFinale_mod[suj]
        mat = scipy.io.loadmat('../MATLAB_DATA/'+name_suj+'/'+name_suj+'-'+nomCond+'timePooled.mat')#pendule pour instant
        data_C3_suj = mat["data"][num_elec]
        liste_C3.append(data_C3_suj)
        for freq in range(82):#all freqs parcourir sujets 
            data_freq_suj[suj][freq] = data_C3_suj[freq]
        
    
    #now estimate the confidence interval point for each freq
    data_freq_mean  = []
    data_lower_cI = []
    data_upper_cI = []
    for freq in range(82-(85-fmax)):
        print("frequence : "+str(3+freq) + " Hz")
        x = data_freq_suj[:,freq]#3Hz
        m = x.mean() 
        s = x.std() 
        dof = 22
        confidence = confidence
        t_crit = np.abs(t.ppf((1-confidence)/2,dof))
        print(len(x))
        data_lower_cI.append((m-s*t_crit/np.sqrt(len(x))))
        data_upper_cI.append((m+s*t_crit/np.sqrt(len(x))))
                             
        print((m-s*t_crit/np.sqrt(len(x)), m+s*t_crit/np.sqrt(len(x))) )
        data_freq_mean.append(np.mean(data_freq_suj[:,freq]))
        
    #now plot this stuff 
    
    freqs = range(3,fmax,1)
    freqs_ticks = range(3,fmax,3)
    
    #fig,ax = plt.subplots()
    ax.plot(freqs, data_freq_mean, '-', color=colorPlot,label=nomCond)
    ax.set_title('Frequency')
    ax.set_xticks(freqs)
    ax.set_xticklabels(freqs)
    ax.set_ylabel('ERD')
    ax.axhline(y=0,color="black",linestyle="dotted")
    ax.axvline(x=8,color="black",linestyle="--")
    ax.axvline(x=30,color="black",linestyle="--")
    ax.set_ylim([-0.45,0.15])
    ax.legend(loc="upper left")
    ax.set_xticks(freqs_ticks)
    ax.fill_between(freqs, data_lower_cI,data_upper_cI , alpha=0.2, color=colorPlot)

# #11 = C3
# fig,ax = plt.subplots()
# plot_oneCond_freqDesync('pendule','tab:orange',fig,ax,55,0.99,11)
# plot_oneCond_freqDesync('main','tab:green',fig,ax,55,0.99,11)
# raw_signal.plot(block=True)

# fig,ax = plt.subplots()
# plot_oneCond_freqDesync('main','tab:green',fig,ax,55,0.99,11)
# plot_oneCond_freqDesync('mainIllusion','tab:blue',fig,ax,55,0.99,11)
# raw_signal.plot(block=True)

# #C3 = 11

# fig, (ax1,ax2,ax3) = plt.subplots(1,3, sharey=True)
# fig.suptitle('C3 95% confidence interval (n=23)')
# plot_oneCond_freqDesync('pendule','tab:orange',fig,ax1,80,0.95,11)
# plot_oneCond_freqDesync('main','tab:green',fig,ax2,80,0.95,11)
# plot_oneCond_freqDesync('mainIllusion','tab:blue',fig,ax3,80,0.95,11)
# fig.tight_layout()

# #C4 = 13
# fig2, (ax1,ax2,ax3) = plt.subplots(1,3, sharey=True)
# fig.suptitle('C4 95% confidence interval (n=23)')
# plot_oneCond_freqDesync('pendule','tab:orange',fig2,ax1,80,0.95,13)
# plot_oneCond_freqDesync('main','tab:green',fig2,ax2,80,0.95,13)
# plot_oneCond_freqDesync('mainIllusion','tab:blue',fig2,ax3,80,0.95,13)
# raw_signal.plot(block=True)

# fig, axs = plt.subplots(2, 3,sharey=True,sharex=True)
# fig.suptitle('C3/C4 95% confidence interval (n=23)')
# plot_oneCond_freqDesync('pendule','tab:orange',fig,axs[0, 0],80,0.95,11)
# plot_oneCond_freqDesync('main','tab:green',fig,axs[0, 1],80,0.95,11)
# plot_oneCond_freqDesync('mainIllusion','tab:blue',fig,axs[0, 2],80,0.95,11)
# plot_oneCond_freqDesync('pendule','tab:orange',fig2,axs[1, 0],80,0.95,13)
# plot_oneCond_freqDesync('main','tab:green',fig2,axs[1, 1],80,0.95,13)
# plot_oneCond_freqDesync('mainIllusion','tab:blue',fig2,axs[1, 2],80,0.95,13)
# raw_signal.plot(block=True)
