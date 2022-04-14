#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:55:18 2022

@author: claire.dussard
"""
#============ puissance selon frequence=====================================================================
import matplotlib.pyplot as plt
import numpy as np
import mne

def compute_freqBand_condition(fmin,fmax,avpower_cond):
    cond_freq = avpower_cond.copy().crop(fmin =fmin,fmax=fmax)
    mean_freq_cond = np.mean(cond_freq.data,axis=1)
    C3_freq_cond = mean_freq_cond[11]
    C4_freq_cond = mean_freq_cond[13]
    return C3_freq_cond,C4_freq_cond

def discretize_time(nbtimePoints,freqEch,C3values,C4values):
    freqEch = freqEch + freqEch/2
    arr_C3 = np.empty(nbtimePoints, dtype=object) 
    arr_C4 = np.empty(nbtimePoints, dtype=object) 
    for i in range(0,nbtimePoints):
        point_temps = C3values[freqEch*i:freqEch*(i+1)].mean()
        arr_C3[i]=point_temps
        point_temps = C4values[freqEch*i:freqEch*(i+1)].mean()
        arr_C4[i]=point_temps
    arrayTime = range(-5,30)
    return arr_C3,arr_C4,arrayTime

def computeC3C4MovingAverage(C3values,C4values):#35 pour tout data, 24 si crop
    arr_C3_ma = computeMovingAverage(C3values,28)
    arr_C4_ma = computeMovingAverage(C4values,28)#en fait non 28 pour tout data, 16 si crop ?
    return arr_C3_ma,arr_C4_ma

def computeMovingAverage(C3values,nvalues):#35 pour tout data, 24 si crop #FONCTION A REVOIR
    arr_C3_movAverage = np.empty(nvalues, dtype=object) 
    compteur_moyenne = 1
    for i in range(1,nvalues):
        print("n value"+str(i))
        if compteur_moyenne == 5:
            print("continue")
            compteur_moyenne += 1
            continue#passe l'instance de la boucle
        elif compteur_moyenne == 6:
            compteur_moyenne = 1
            print("continue")
            continue
        offset = 125*i
        point_1 = C3values[250*i :250*(i+1) ].mean()
        point_2 = C3values[63+(250*i):63 + (250*(i+1))].mean()
        point_3 = C3values[125+(250*i) :125 +(250*(i+1))].mean()
        point_4 = C3values[188+(250*i) :188 +(250*(i+1)) ].mean()
        pointMoyenne = (point_1+point_2+point_3 + point_4)/4
        arr_C3_movAverage[i] = pointMoyenne
        compteur_moyenne += 1
    return arr_C3_movAverage

#yo = computeMovingAverage(C3_freq_main,28) 
  
def computeSlidingWindowEpochs(C3_freq,nvalues):
    arr_C3_movAverage_freq = np.zeros(shape=(C3_freq.shape[0],nvalues))
    offset = 0 #pour gerer le fait qu'on jette deux fenetres d'1s entre les fenetres : ajouter 0.5s apres chaque FB (125pts a 250Hz)
    for i in range(0,nvalues):
        for j in range(C3_freq.shape[0]):#freqs
            point_1 = C3_freq[j,250*i:250*(i+1)].mean()
            point_2 = C3_freq[j,63+(250*i):63 + (250*(i+1))].mean()
            point_3 = C3_freq[j,125+(250*i):125 +(250*(i+1))].mean()
            point_4 = C3_freq[j,188+(250*i):188 +(250*(i+1))].mean()
            tableauMoyenne = (point_1+point_2+point_3 + point_4)/4
            arr_C3_movAverage_freq[j,i] = tableauMoyenne
    return arr_C3_movAverage_freq
     #mne.make_fixed_length_epochs(raw, duration=1.0, preload=False, reject_by_annotation=True, proj=True, overlap=0.0, id=1, verbose=None
    
def computeSlidingWindowEpochs_elecFreq(C3_freq_elec,nvalues):
    n_elec = C3_freq_elec.shape[0]
    n_freqs = C3_freq_elec.shape[1]
    arr_C3_movAverage_electrode_freq = np.zeros(shape=(n_elec,n_freqs,nvalues))
    offset = 0 
    for tps in range(0,nvalues):
        for elec in range(n_elec):#electrodes
            for freq in range(n_freqs):#freqs
                point_1 = C3_freq_elec[elec,freq,250*tps+ offset:250*(tps+1)+ offset].mean()
                point_2 = C3_freq_elec[elec,freq,63+(250*tps)+ offset:63 + (250*(tps+1))+ offset].mean()
                point_3 = C3_freq_elec[elec,freq,125+(250*tps)+ offset:125 +(250*(tps+1))+ offset].mean()
                point_4 = C3_freq_elec[elec,freq,188+(250*tps)+ offset:188 +(250*(tps+1))+ offset].mean()
                tableauMoyenne = (point_1+point_2+point_3 + point_4)/4
                arr_C3_movAverage_electrode_freq[elec,freq,tps] = tableauMoyenne
    return arr_C3_movAverage_electrode_freq
  
    
def plot_freqBand(fmin,fmax,avpower_main,avpower_mainIllusion,avpower_pendule,fig, ax1,ax2,times,ymin,ymax):#0.28e-11,0.5e-11
    C3_freq_main,C4_freq_main = compute_freqBand_condition(fmin,fmax,avpower_main)
    #arr_C3_main,arr_C4_main = computeC3C4MovingAverage(C3_freq_main,C4_freq_main)
    #arr_C3_main,arr_C4_main,arrayTime = discretize_time(35,250,C3_freq_main,C4_freq_main)
    C3_freq_mainIllusion,C4_freq_mainIllusion = compute_freqBand_condition(fmin,fmax,avpower_mainIllusion)
    #arr_C3_mainIllusion,arr_C4_mainIllusion = computeC3C4MovingAverage(C3_freq_mainIllusion,C4_freq_mainIllusion)
    #arr_C3_mainIllusion,arr_C4_mainIllusion,arrayTime = discretize_time(35,250,C3_freq_mainIllusion,C4_freq_mainIllusion)
    C3_freq_pendule,C4_freq_pendule = compute_freqBand_condition(fmin,fmax,avpower_pendule)
    #arr_C3_pendule,arr_C4_pendule = computeC3C4MovingAverage(C3_freq_pendule,C4_freq_pendule)
    #arr_C3_pendule,arr_C4_pendule,arrayTime = discretize_time(35,250,C3_freq_pendule,C4_freq_pendule)
    #times=range(-5,30)
    #start_times =[-4.75 + (1.25)*i for i in range(28)]#ou 35
    #print(start_times)
    #times = [start + 0.5 for start in start_times]#milieu de fenetre
    print(times)
    arr_C3_main,arr_C4_main = C3_freq_main,C4_freq_main#Si pas de moyenne tps
    arr_C3_mainIllusion,arr_C4_mainIllusion = C3_freq_mainIllusion,C4_freq_mainIllusion
    arr_C3_pendule,arr_C4_pendule = C3_freq_pendule,C4_freq_pendule
    ax1.set_ylim([ymin,ymax])#ax1.set_ylim([-0.35, 0.23])# pb d'echelle pour tout afficher ensemble car 1/f power sans BL
    ax1.plot(times,arr_C3_pendule,label="pendule")
    ax1.plot(times,arr_C3_main,label="main")
    ax1.plot(times,arr_C3_mainIllusion,label="mainIllusion")
    ax1.axvline(0, color='black',ls=':')
    ax1.axvline(1.5, color='saddlebrown',ls='--')
    ax1.axvline(26.9, color='saddlebrown',ls='--')
    ax1.set_title("C3 : "+str(fmin)+"-"+str(fmax)+'Hz')
    ax1.legend(loc='lower right')
    #afficher vibrations
    ax1.axvline(6.5, color='black', linestyle='--')
    ax1.axvline(8.3, color='black', linestyle='--')
    ax1.axvline(12.7, color='black', linestyle='--')
    ax1.axvline(14.5, color='black', linestyle='--')
    ax1.axvline(18.92, color='black', linestyle='--')
    ax1.axvline(20.72, color='black', linestyle='--')
    ax1.axvline(25.22, color='black', linestyle='--')
    ax1.axvline(27.02, color='black', linestyle='--')
    #fin vib
    ax2.set_ylim([ymin,ymax])#ax2.set_ylim([-0.35, 0.23])
    ax2.plot(times,arr_C4_pendule,label="pendule")
    ax2.plot(times,arr_C4_main,label="main")
    ax2.plot(times,arr_C4_mainIllusion,label="mainIllusion")
    ax2.axvline(0, color='black',ls=':')
    ax2.axvline(1.5, color='saddlebrown',ls='--')
    ax2.axvline(26.9, color='saddlebrown',ls='--')
    #afficher vibrations
    ax2.axvline(6.5, color='black', linestyle='--')
    ax2.axvline(8.3, color='black', linestyle='--')
    ax2.axvline(12.7, color='black', linestyle='--')
    ax2.axvline(14.5, color='black', linestyle='--')
    ax2.axvline(18.92, color='black', linestyle='--')
    ax2.axvline(20.72, color='black', linestyle='--')
    ax2.axvline(25.22, color='black', linestyle='--')
    ax2.axvline(27.02, color='black', linestyle='--')
    #fin vib
    ax2.set_title("C4 : "+str(fmin)+"-"+str(fmax)+'Hz')
    ax2.legend(loc='lower right')
    return fig,arr_C3_pendule,arr_C3_main,arr_C3_mainIllusion,arr_C4_pendule,arr_C4_main,arr_C4_mainIllusion



def plot_condition(avpower_cond,fig, ax1,ax2,times,nomCondition,ymin,ymax):
    ax1.set_ylim([ymin,ymax])#ax1.set_ylim([-0.35, 0.3])#1.65e-11
    ax2.set_ylim([ymin,ymax])#ax2.set_ylim([-0.35, 0.3])#1.65e-11
    C3_freq_cond,C4_freq_cond = compute_freqBand_condition(3,7,avpower_cond)
    #arr_C3_freq,arr_C4_freq = computeC3C4MovingAverage(C3_freq_cond,C4_freq_cond)
    #arr_C3_freq,arr_C4_freq,arrayTime = discretize_time(35,250,C3_freq_cond,C4_freq_cond)
    #times = arrayTime
    arr_C3_freq = C3_freq_cond
    arr_C4_freq = C4_freq_cond
    #times = range(-5,30,9000)
    #start_times =[-5 + (1.25)*i for i in range(24)]#ou 35
    #print(start_times)
    #times = [start + 0.5 for start in start_times]#milieu de fenetre
    print(times)
    ax1.plot(times,arr_C3_freq,label="3-7Hz")
    ax2.plot(times,arr_C4_freq,label="3-7Hz")
    C3_freq_cond,C4_freq_cond = compute_freqBand_condition(8,13,avpower_cond)
    #arr_C3_freq,arr_C4_freq = computeC3C4MovingAverage(C3_freq_cond,C4_freq_cond)
    arr_C3_freq = C3_freq_cond
    arr_C4_freq = C4_freq_cond    
    #arr_C3_freq,arr_C4_freq,arrayTime = discretize_time(35,250,C3_freq_cond,C4_freq_cond)
    ax1.plot(times,arr_C3_freq,label="8-13Hz")
    ax2.plot(times,arr_C4_freq,label="8-13Hz")
    C3_freq_cond,C4_freq_cond = compute_freqBand_condition(13,20,avpower_cond)
    #arr_C3_freq,arr_C4_freq = computeC3C4MovingAverage(C3_freq_cond,C4_freq_cond)
    #arr_C3_freq,arr_C4_freq,arrayTime = discretize_time(35,250,C3_freq_cond,C4_freq_cond)
    arr_C3_freq = C3_freq_cond
    arr_C4_freq = C4_freq_cond
    ax1.plot(times,arr_C3_freq,label="13-20Hz")
    ax2.plot(times,arr_C4_freq,label="13-20Hz")
    C3_freq_cond,C4_freq_cond = compute_freqBand_condition(20,30,avpower_cond)
    arr_C3_freq = C3_freq_cond
    arr_C4_freq = C4_freq_cond
    #arr_C3_freq,arr_C4_freq = computeC3C4MovingAverage(C3_freq_cond,C4_freq_cond)
    #arr_C3_freq,arr_C4_freq,arrayTime = discretize_time(35,250,C3_freq_cond,C4_freq_cond)
    ax1.plot(times,arr_C3_freq,label="20-30Hz")
    ax2.plot(times,arr_C4_freq,label="20-30Hz")
    # C3_freq_cond,C4_freq_cond = compute_freqBand_condition(8,30,avpower_cond)
    # ax1.plot(times,C3_freq_cond,label="8-30Hz")
    # ax2.plot(times,C4_freq_cond,label="8-30Hz")
    C3_freq_cond,C4_freq_cond = compute_freqBand_condition(12,15,avpower_cond)
    ax1.plot(times,C3_freq_cond,label="12-15Hz")
    ax2.plot(times,C4_freq_cond,label="12-15Hz")
    ax1.set_title("C3 : "+ nomCondition)
    ax2.set_title("C4 : "+ nomCondition)
    ax1.axvline(0, color='black',ls=':')
    ax1.axvline(1.5, color='saddlebrown',ls='--')
    ax1.axvline(26.9, color='saddlebrown',ls='--')
    ax2.axvline(0, color='black',ls=':')
    ax2.axvline(1.5, color='saddlebrown',ls='--')
    ax2.axvline(26.9, color='saddlebrown',ls='--')
    #ajout pr vibrations
    ax1.axvline(6.5, color='black', linestyle='--')
    ax1.axvline(8.3, color='black', linestyle='--')
    ax1.axvline(12.7, color='black', linestyle='--')
    ax1.axvline(14.5, color='black', linestyle='--')
    ax1.axvline(18.92, color='black', linestyle='--')
    ax1.axvline(20.72, color='black', linestyle='--')
    ax1.axvline(25.22, color='black', linestyle='--')
    ax1.axvline(27.02, color='black', linestyle='--')
    #fin ajout pr vibrations

    ax2.axvline(6.5, color='black', linestyle='--')
    ax2.axvline(8.3, color='black', linestyle='--')
    ax2.axvline(12.7, color='black', linestyle='--')
    ax2.axvline(14.5, color='black', linestyle='--')
    ax2.axvline(18.92, color='black', linestyle='--')
    ax2.axvline(20.72, color='black', linestyle='--')
    ax2.axvline(25.22, color='black', linestyle='--')
    ax2.axvline(27.02, color='black', linestyle='--')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    

def plot_allfreqBand_groupByFrequency(avpower_main,avpower_mainIllusion,avpower_pendule,vmin,vmax):
    times = avpower_main.times #times passed si pas de moyennage tps
    fig, axs = plt.subplots(4, 2)#autant de lignes que bandes de freq, autant de colonnes que d'electrodes  #les minmax sont individuels si data sans BL
    #res_3_7 = plot_freqBand(3,7,avpower_main,avpower_mainIllusion,avpower_pendule,fig,axs[0,0],axs[0,1],times,vmin,vmax)#3.4e-11,1.3e-10)#alpha
    res_8_13 = plot_freqBand(8,13,avpower_main,avpower_mainIllusion,avpower_pendule,fig,axs[0,0],axs[0,1],times,vmin,vmax)#2.5e-11,7e-11)#alpha
    res_12_15 = plot_freqBand(12,15,avpower_main,avpower_mainIllusion,avpower_pendule,fig,axs[1,0],axs[1,1],times,vmin,vmax)#1.1e-11,5.5e-11)#high beta
    res_13_20 = plot_freqBand(13,20,avpower_main,avpower_mainIllusion,avpower_pendule,fig,axs[2,0],axs[2,1],times,vmin,vmax)#1e-11,5e-11)#low beta
    res_20_30 = plot_freqBand(20,30,avpower_main,avpower_mainIllusion,avpower_pendule,fig,axs[3,0],axs[3,1],times,vmin,vmax)#1.1e-11,5.5e-11)#high beta
    #res_8_30 = plot_freqBand(8,30,avpower_main,avpower_mainIllusion,avpower_pendule,fig,axs[3,0],axs[3,1],times,vmin,vmax)#1.0e-11,4.5e-11)#high beta
    #plot_freqBand(30,50,avpower_main,avpower_mainIllusion,avpower_pendule,fig,axs[4,0],axs[4,1])#alpha
    #plot_freqBand(50,80,avpower_main,avpower_mainIllusion,avpower_pendule,fig,axs[5,0],axs[5,1])#alpha
    for ax in axs.flat:
        ax.set(ylabel='Bandpower')
    axs[3,0].set(xlabel='Time(s)')
    axs[3,1].set(xlabel='Time(s)')

    return None#res_3_7,res_8_13,res_13_20,res_8_30#,res_13_20,res_20_30,



# res_8_13,res_13_20,res_20_30,res_8_30 = plot_allfreqBand_groupByFrequency(av_power_main,av_power_mainIllusion,av_power_pendule)

# raw_signal.plot(block=True)

def plot_allfreqBand_groupByCondition(avpower_main,avpower_mainIllusion,avpower_pendule,vmin,vmax):
    times = avpower_main.times
    fig, axs = plt.subplots(3, 2)#autant de lignes que de conditions, autant de colonnes que d'electrodes
    plot_condition(avpower_pendule,fig, axs[0,0],axs[0,1],times,"pendule",vmin,vmax)#6e-12,10e-11)
    plot_condition(avpower_main,fig, axs[1,0],axs[1,1],times,"main",vmin,vmax)#6e-12,10e-11)
    plot_condition(avpower_mainIllusion,fig, axs[2,0],axs[2,1],times,"mainIllusion",vmin,vmax)#6e-12,10e-11)
    for ax in axs.flat:
        ax.set(ylabel='Bandpower')
    axs[2,0].set(xlabel='Time(s)')
    axs[2,1].set(xlabel='Time(s)')
    return True

# plot_allfreqBand_groupByCondition(av_power_main,av_power_mainIllusion,av_power_pendule)
# av_power_mainIllusion_noBL_seuil =  mne.time_frequency.read_tfrs("../withoutBaseline/mainIllusionSeuil_mean-tfr.h5")[0]

# plot_allfreqBand_groupByCondition(av_power_main_noBL_seuil,av_power_mainIllusion_noBL_seuil,av_power_pendule_noBL_seuil)
# plot_allfreqBand_groupByFrequency(av_power_main_noBL_seuil,av_power_mainIllusion_noBL_seuil,av_power_pendule_noBL_seuil)
# raw_signal.plot(block=True)

#laplacien
# av_power_mainIllusion_C3laplacien_seuil = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion_C3C4laplacien_noBL_seuil-tfr.h5")[0]
# av_power_main_C3laplacien_seuil = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main_C3C4laplacien_noBL_seuil-tfr.h5")[0]
# av_power_pendule_C3laplacien_seuil = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule_C3C4laplacien_noBL_seuil-tfr.h5")[0]
#on redefinit la fonction car on a que 2 channels laplaciens virtuels
def compute_freqBand_condition_laplacien(fmin,fmax,avpower_cond):
    print("VERSION LAPLACIEN compute_freqBand_condition")
    cond_freq = avpower_cond.copy().crop(fmin =fmin,fmax=fmax)
    mean_freq_cond = np.mean(cond_freq.data,axis=1)
    C3_freq_cond = mean_freq_cond[0,:]
    C4_freq_cond = mean_freq_cond[1,:]
    return C3_freq_cond,C4_freq_cond

# plot_allfreqBand_groupByFrequency(av_power_main_C3laplacien_seuil,av_power_mainIllusion_C3laplacien_seuil,av_power_pendule_C3laplacien_seuil)
# raw_signal.plot(block=True)

# plot_allfreqBand_groupByCondition(av_power_main_C3laplacien_seuil,av_power_mainIllusion_C3laplacien_seuil,av_power_pendule_C3laplacien_seuil)
# raw_signal.plot(block=True)
