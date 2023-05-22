#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:43:50 2021

@author: claire.dussard
"""
import os 
import mne
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import pathlib
import mne
import numpy as np
#create list with all the paths to feedback condition recordings
def createListeCheminsSignaux(essaisFeedback,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates):#essaisPendule#essaisPendule#essaisMainSeule#essaisMainTactile
    #ex : pour tous essais mainSeule
    liste_rawPath = []
    listeEssais = essaisFeedback
    for num_sujet in allSujetsDispo:
        print("sujet n° "+str(num_sujet))
        print("nom essai"+listeEssais[num_sujet])
        if len(listeEssais[num_sujet])>5:
            print(listeEssais[num_sujet])
            pass#pb condition manquante
        else:     
            nom_sujet = listeNumSujetsFinale[num_sujet]
            if num_sujet in SujetsPbNomFichiers:
                print("correcting date")
                if num_sujet>0:
                    date_sujet = '19-04-2021'#modifier le nom du fichier ne suffit pas, la date est ds les vhdr
                else:
                    date_sujet = '15-04-2021'
            else:
                date_sujet = dates[num_sujet]
            date_nom_fichier = date_sujet[-4:]+"-"+date_sujet[3:5]+"-"+date_sujet[0:2]+"_"
            dateSession = listeDatesFinale[num_sujet]
            sample_data_loc = nom_sujet+"/"+listeDatesFinale[num_sujet]+"/eeg"
            sample_data_dir = pathlib.Path(sample_data_loc)
            nom_essai = listeEssais[num_sujet]
            print(nom_essai) #contient l'info de s'il faut prendre 5-2,6-2-c etc
            raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + nom_essai+".vhdr")#il faudrait recup les epochs et les grouper ?
            liste_rawPath.append(raw_path_sample)
            
    print(len(liste_rawPath))#cb d'enregistrements recuperes
    return liste_rawPath

def plotSave_power_topo_cond(epochData_cond,listeRawPath_cond,freqMin,freqMax,nomCond,downSampleFreq,tmin,tmax,allSujetsDispo): # compute, plot & save images for individual & all subjects data
    import matplotlib.pyplot as plt    
    #====================individual data=============================
    i = 0
    freqs = np.arange(freqMin, freqMax, 1)  # frequencies from 2-35Hz
    n_cycles = freqs 
    dureePreBaseline = 3.0
    dureePreBaseline = - dureePreBaseline
    dureeBaseline = 2.0
    valeurPostBaseline = dureePreBaseline + dureeBaseline
    mode = 'logratio'
    liste_power_sujets = []
    for epochData_cond_sujet,rawPath_cond_sujet in zip(epochData_cond,listeRawPath_cond):
        print("\n===========Sujet S "+str(allSujetsDispo[i])+"========================\n")#a remplacer si on fait des subsets de sujets
        epochData_cond_sujet_down = epochData_cond_sujet.resample(downSampleFreq, npad='auto') 
        #compute power
        print("computing power...")
        #DECIM VAUT 1
        power_sujet = mne.time_frequency.tfr_morlet(epochData_cond_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False)#,return_itc=False, decim=3, n_jobs=1)
        #compute topomaps
        path_sujet = listeRawPath_cond[i]
        print("computing & saving individual topomaps...")
        save_topo_data(power_sujet,dureePreBaseline,valeurPostBaseline,path_sujet,mode,nomCond,True,tmin,tmax,i)#attention, le implique de ne pas utiliser de sous liste de sujet
        liste_power_sujets.append(power_sujet)
        i +=1
    return liste_power_sujets

ScalesSujetsGraphes_8a30Hz = [0.2,0.3,0.38,0.28,0.4,#S00-06
                              0.28,0.28,0.3,0.4,0.34,#S07-11
                              0.18,0.24,0.35,0.25,0.32,#S12-16
                              0.22,0.4,0.24,0.16,0.3,#S17-21
                              0.4,0.26,0.4,0.2,0.09]#S22-24 + echelle generale + echelle difference

ScalesSujetsGraphes_Theta = [0.24,0.28,0.3,0.24,0.24,#S00-06
                             0.28,0.24,0.28,0.33,0.2,#S07-11
                             0.2,0.3,0.28,0.24,0.28,#S12-16
                             0.22,0.24,0.18,0.22,0.28,#S17-21
                             0.24,0.22,0.18,0.12,0.09]#S22-24 + echelle generale

ScalesSujetsGraphes_Alpha = [0.44,0.32,0.38,0.25,0.34,#S00-06
                             0.26,0.32,0.38,0.4,0.34,#S07-11
                             0.18,0.3,0.35,0.26,0.38,#S12-16
                             0.24,0.28,0.3,0.28,0.4,#S17-21
                             0.4,0.4,0.35,0.28,0.09]#S22-24 + echelle generale

ScalesSujetsGraphes_LowBeta = [0.28,0.3,0.38,0.28,0.38,#S00-06
                               0.32,0.3,0.33,0.4,0.3,#S07-11
                               0.32,0.24,0.38,0.32,0.34,#S12-16
                               0.25,0.38,0.32,0.2,0.35,#S17-21
                               0.4,0.28,0.35,0.22,0.09]#S22-24 + echelle generale

ScalesSujetsGraphes_HighBeta = [0.28,0.3,0.3,0.3,0.4,#S00-06
                                0.3,0.3,0.3,0.33,0.38,#S07-11
                                0.28,0.35,0.38,0.35,0.25,#S12-16
                                0.28,0.4,0.24,0.27,0.32,#S17-21
                                0.36,0.28,0.35,0.2,0.09]#S22-24 + echelle generale

ScalesSujetsGraphes_LowGamma = [0.24,0.35,0.25,0.3,0.42,#S00-06
                                0.3,0.3,0.32,0.3,0.4,#S07-11
                                0.3,0.38,0.3,0.38,0.28,#S12-16
                                0.26,0.45,0.38,0.26,0.32,#S17-21
                                0.28,0.32,0.28,0.2,0.09]#S22-24 + echelle generale

ScalesSujetsGraphes_HighGamma = [0.24,0.35,0.25,0.3,0.42,#S00-06
                                 0.3,0.3,0.36,0.3,0.4,#S07-11
                                 0.38,0.4,0.3,0.38,0.28,#S12-16
                                 0.28,0.45,0.4,0.26,0.34,#S17-21
                                 0.34,0.38,0.25,0.2,0.09]#S22-24 + echelle generale

#save carte topo data
def save_topo_data(power,dureePreBaseline,valeurPostBaseline,path_sujet,mode,nomCond,doBaseline,tmin,tmax,i):#can be improved with boolean Params for alpha etc
    my_cmap = discrete_cmap(13, 'RdBu_r')
    if doBaseline ==False:
        print("no baseline")
        baseline = None
    elif doBaseline == True:
        baseline = (dureePreBaseline,valeurPostBaseline)
    if path_sujet!= "all_sujets":                                                    #and having parameters for the max values of the graphs
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split('/')
        directory = "../images/" + path_raccourci_split[0] + "/" 
    else:
        directory = "../images/"+path_sujet+"/"
        path_raccourci_split = ["all_sujets"]
    #check if directory exists
    if not os.path.exists(directory): 
        os.makedirs(directory)      #1.5 tmin 25.5 tmax
        #scales
#     ScalesSujetsGraphes_8a30Hz = [0.2,0.3,0.32,0.28,0.35,0.28,0.28,0.3,0.35,0.32,0.2,0.2,0.32,0.22,0.34,0.25,0.35,0.26,0.2,0.27,0.35,0.26,-1,0.32,0.12]#8-30Hz
#     ScalesSujetsGraphes_Theta = [0.24,0.3,0.3,0.24,0.24,0.24,0.24,0.26,0.28,0.22,0.22,0.28,0.28,0.26,0.28,0.22,0.24,0.2,0.22,0.26,0.22,0.22,-1,0.2,0.1]#3-7Hz
#     ScalesSujetsGraphes_Alpha = [0.4,0.32,0.35,0.25,0.3,0.28,0.32,0.32,0.35,0.32,0.18,0.3,0.32,0.26,0.34,0.22,0.28,0.28,0.28,0.34,0.35,0.35,-1,0.3,0.1]#8-13Hz
#     ScalesSujetsGraphes_LowBeta = [0.24,0.3,0.35,0.28,0.35,0.3,0.3,0.3,0.35,0.28,0.3,0.24,0.34,0.34,0.34,0.22,0.35,0.28,0.22,0.32,0.35,0.28,-1,0.32,0.1]#13-20Hz
#     ScalesSujetsGraphes_HighBeta = [0.28,0.3,0.3,0.3,0.35,0.3,0.3,0.3,0.3,0.35,0.28,0.32,0.35,0.34,0.25,0.28,0.35,0.26,0.27,0.32,0.34,0.28,-1,0.32,0.1]#20-30Hz
#     ScalesSujetsGraphes_LowGamma = [0.24,0.32,0.25,0.3,0.36,0.3,0.3,0.32,0.3,0.35,0.3,0.32,0.28,0.35,0.28,0.28,0.38,0.34,0.26,0.32,0.25,0.32,-1,0.26,0.1]#30-50Hz
#     ScalesSujetsGraphes_HighGamma = [0.24,0.32,0.25,0.3,0.36,0.3,0.3,0.32,0.3,0.35,0.35,0.35,0.28,0.35,0.28,0.28,0.38,0.35,0.26,0.32,0.32,0.34,-1,0.26,0.1]#50-80Hz
# #======================Computing all powers and saving=================    
    scale_8a30Hz = ScalesSujetsGraphes_8a30Hz[i]
    scale_theta = ScalesSujetsGraphes_Theta[i]
    scale_alpha = ScalesSujetsGraphes_Alpha[i]
    scale_lowBeta = ScalesSujetsGraphes_LowBeta[i]
    scale_highBeta = ScalesSujetsGraphes_HighBeta[i]
    scale_lowGamma = ScalesSujetsGraphes_LowGamma[i]
    scale_highGamma = ScalesSujetsGraphes_HighGamma[i] #MODIFIE POUR QUE SMR, A REMETTRE EN DECOMMENTANT SI BESOIN
    fig = power.plot_topomap(baseline=baseline,mode=mode,fmin = 8,fmax=30,tmin=tmin,tmax=tmax,vmin=-scale_8a30Hz,vmax=scale_8a30Hz,cmap=my_cmap)#8-30Hz
    fig.set_size_inches(25.5, 12.5)
    plt.savefig(directory+ path_raccourci_split[0] +"-"+nomCond+"_8-30Hz.png",bbox_inches='tight',transparent=True)#save components picture
    plt.close(fig)
    return True


#save ICA data
def save_ICA_files(liste_ICA,liste_rawPathmodif,windows):
    import matplotlib.pyplot as plt
#Save files sauvegarde ICA liste_rawPathmodif=liste_rawPath[21:24]
    if windows:
        charac_split = "\\"
    else:
        charac_split = "/"
    i=0
    for ica_to_save in liste_ICA:
        path_sujet = liste_rawPathmodif[i]
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../ICA/" + path_raccourci_split[0] + "/" 
        fig = ica_to_save.plot_components(picks=['eeg'], show=False, verbose=False)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(directory+ path_raccourci_split[3] +"png")#save components picture
        plt.close(fig)
        ica_to_save.save(directory+ path_raccourci_split[3][:-1] +"-ica.fif",overwrite=True)#save ICA computed
        i += 1 
    print("done saving")
    return True

def load_ICA_files_windows(liste_rawPath,windows):
    liste_ICA = []
    if windows:
        charac_split = "\\"
    else:
        charac_split = "/"
    for rawPath in liste_rawPath:
        path_raccourci = str(rawPath)[0:len(str(rawPath))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../ICA/" + path_raccourci_split[0] + "/" 
        fname = directory+ path_raccourci_split[3][:-1] + "-ica.fif"
        ica_i = mne.preprocessing.read_ica(fname, verbose=None)
        liste_ICA.append(ica_i)
    return liste_ICA

#save epochs after ICA
def saveEpochsAfterICA_apresdropBad_windows(listeEpochs,liste_rawPath,windows):    #Save files epochs_ICaises
    i=0
    if windows:
        charac_split = "\\"
    else:
        charac_split = "/"
    for signal in listeEpochs:
        if signal is not None:
            path_sujet = liste_rawPath[i]#attention ne marche que si on a les epochs dans l'ordre
            path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
            path_raccourci_split = path_raccourci.split(charac_split)
            directory = "../EPOCH_ICA_APRES_REF/" + path_raccourci_split[0] + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            signal.save(directory+ path_raccourci_split[3] +"fif",overwrite=True)
        else:
            print("signal is NONE")
        i += 1
            
    print("done saving")
    return True

def saveEpochsAfterICA_avantdropBad_windows(listeEpochs,liste_rawPath,windows):    #Save files epochs_ICaises
    i=0
    if windows:
        charac_split = "\\"
    else:
        charac_split = "/"
    for signal in listeEpochs:
        path_sujet = liste_rawPath[i]#attention ne marche que si on a les epochs dans l'ordre
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../EPOCH_ICA_avant_dropBad_avant_averageRef/" + path_raccourci_split[0] + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        signal.save(directory+ path_raccourci_split[3] +"fif",overwrite=True)
        i += 1
    print("done saving")
    return True



def saveEpochsAfterICA_FBseul(listeEpochs,liste_rawPath,nomCond,windows):    #Save files epochs_ICaises
    nomCond = "-" + nomCond
    i=0
    if windows:
        charac_split = "\\"
    else:
        charac_split = "/"
        
    for signal in listeEpochs:
        path_sujet = liste_rawPath[i]#attention ne marche que si on a les epochs dans l'ordre
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../EPOCH_ICA_APRES_REF/" + path_raccourci_split[0] + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        signal.save(directory+ path_raccourci_split[3][:-1]+ nomCond+".fif",overwrite=True)
        i += 1
    print("done saving")
    return True


def save_tfr_data(listeAverageTFR,listerawPath,suffixe,windows):
    if windows:
        charac_split = "\\"
    else:
        charac_split = "/"
    if suffixe != "":
        suffixe = "-" + suffixe
    i = 0
    for averageTFR in listeAverageTFR:
        print("saving subj"+str(i))
        path_sujet = listerawPath[i]#attention ne marche que si on a les epochs dans l'ordre
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../AV_TFR/" + path_raccourci_split[0] + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        averageTFR.save(directory+ path_raccourci_split[3][:-1] + suffixe + "-tfr.h5",overwrite=True)#TO DO pop average TFR to free memory
        i += 1
    print("done saving")
    return True

import scipy
from scipy import io
def save_tfr_data_to_mat(listeAverageTFR,listerawPath,suffixe,doBaseline,baseline,tmin,tmax):
    if suffixe != "":
        suffixe = "-" + suffixe
    i = 0
    for averageTFR in listeAverageTFR:
        print("saving subj"+str(i))
        path_sujet = listerawPath[i]#attention ne marche que si on a les epochs dans l'ordre
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split('/')
        directory = "../MATLAB_DATA/" + path_raccourci_split[0] + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if doBaseline:
            averageTFR.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
        averageTFR.crop(tmin=tmin,tmax=tmax)
        io.savemat(directory+ path_raccourci_split[0] + suffixe+".mat", {'data': averageTFR.data })
        i += 1
    print("done saving")
    return True

def save_elec_fr_data_to_mat(listeAverageTFR,listerawPath,suffixe,doBaseline,baseline,tmin,tmax,windows):
    if windows:
        charac_split = "\\"
    else:
        charac_split = "/"
    if suffixe != "":
        suffixe = "-" + suffixe
    i = 0
    for averageTFR in listeAverageTFR:
        print("saving subj"+str(i))
        path_sujet = listerawPath[i]#attention ne marche que si on a les epochs dans l'ordre
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../MATLAB_DATA/" + path_raccourci_split[0] + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if doBaseline:
            averageTFR.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
        averageTFR.crop(tmin=tmin,tmax=tmax)
        #mean over time 
        averageTFR_withoutTime = np.mean(averageTFR.data,axis=2)
        scipy.io.savemat(directory+ path_raccourci_split[0] + suffixe+"timePooled"+".mat", {'data': averageTFR_withoutTime.data })
        i += 1
    print("done saving")
    return True

        
def load_tfr_data_windows(liste_rawPath,suffixe,windows):
    if windows:
        charac_split = "\\"
    else:
        charac_split = "/"
    if suffixe != "":
        suffixe = "-" + suffixe
    liste_tfr = []
    i = 0
    for path in liste_rawPath:
        print("path")
        print(path)
        path_sujet = liste_rawPath[i]#attention ne marche que si on a les epochs dans l'ordre
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../AV_TFR/" + path_raccourci_split[0] + "/"
        print("directory")
        print(directory)
        print(directory+ path_raccourci_split[3][:-1] +suffixe+"-tfr.h5")
    
        if os.path.exists(directory):
             try:
                 signal =  mne.time_frequency.read_tfrs(directory+ path_raccourci_split[3][:-1] +suffixe+"-tfr.h5")
             except OSError as e:
                 print(e.errno)
        else:
             print("sujet "+str(i)+" non traité")
        
        liste_tfr.append(signal)
        i += 1
    liste_tfr = [tfr[0] for tfr in liste_tfr] 
    return liste_tfr

def load_tfr_data(liste_rawPath,suffixe):
    if suffixe != "":
        suffixe = "-" + suffixe
    liste_tfr = []
    i = 0
    for path in liste_rawPath:
        path_sujet = liste_rawPath[i]#attention ne marche que si on a les epochs dans l'ordre
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split('/')
        directory = "../AV_TFR/" + path_raccourci_split[0] + "/"
        print(directory)
        if os.path.exists(directory):
            try:
                signal =  mne.time_frequency.read_tfrs(directory+ path_raccourci_split[3][:-1] +suffixe+"-tfr.h5")
            except OSError as e:
                print(e.errno)
        else:
            print("sujet "+str(i)+" non traité")
        
        liste_tfr.append(signal)
        i += 1
    liste_tfr = [tfr[0] for tfr in liste_tfr] 
    return liste_tfr
    

def load_data_postICA_postdropBad_windows(liste_rawPath,suffixe,windows):
    if windows:
        charac_split = "\\"
    else:
        charac_split = "/"
    if suffixe != "":
        suffixe = "-" + suffixe
    import os
    liste_signaux_loades = []   
    i = 0
    for path in liste_rawPath:
        path_sujet = liste_rawPath[i]
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../EPOCH_ICA_APRES_REF/" + path_raccourci_split[0] + '/'
        print(directory)
        if os.path.exists(directory):
            try:
                signal = mne.read_epochs(directory+ path_raccourci_split[3][:-1]+suffixe +".fif")
            except OSError as e:
                print(e.errno)
        else:
            print("sujet "+str(i)+" non traité")
        liste_signaux_loades.append(signal)
        i += 1
    return liste_signaux_loades

def load_data_postICA_preDropbad(liste_rawPath,suffixe,windows):
    if windows:
        charac_split = "\\"
    else:
        charac_split = "/"
    if suffixe != "":
        suffixe = "-" + suffixe
    import os
    liste_signaux_loades = []   
    i = 0
    for path in liste_rawPath:
        path_sujet = liste_rawPath[i]
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../EPOCH_ICA_avant_dropBad_avant_averageRef/" + path_raccourci_split[0] + "/"
        print(directory)
        if os.path.exists(directory):
            try:
                signal = mne.read_epochs(directory+ path_raccourci_split[3][:-1]+suffixe +".fif")
            except OSError as e:
                print(e.errno)
        else:
            print("sujet "+str(i)+" non traité")
        liste_signaux_loades.append(signal)
        i += 1
    return liste_signaux_loades

def load_data_postICA_preDropbad_effetFBseul(liste_rawPath,suffixe,windows,avantDropBad):
    if avantDropBad:
        dossier = "EPOCH_ICA_avant_dropBad_avant_averageRef/"
    else:
        dossier = "EPOCH_ICA_APRES_REF/"
    if windows:
        charac_split = "\\"
    else:
        charac_split = "/"
    if suffixe != "":
        suffixe = "-" + suffixe
    import os
    liste_signaux_loades = []   
    i = 0
    for path in liste_rawPath:
        path_sujet = liste_rawPath[i]
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../"+dossier + path_raccourci_split[0] + "/"
        print(directory)
        if os.path.exists(directory):
            try:
                signal = mne.read_epochs(directory+ path_raccourci_split[3][:-1]+suffixe +".fif")
            except OSError as e:
                print(e.errno)
        else:
            print("sujet "+str(i)+" non traité")
        liste_signaux_loades.append(signal)
        i += 1
    return liste_signaux_loades


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def discrete_log_cmap(N, base_cmap=None):#marche tres bof
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.logspace(0.9, 0.0001, N,base=0.01))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize, \
    LogNorm
import numpy as np
def cmap_discrete(cmap_list):#vole de panda power

    cmap_colors = []
    boundaries = []
    last_upper = None
    for (lower, upper), color in cmap_list:
        if last_upper is not None and lower != last_upper:
            raise ValueError("Ranges for colormap must be continuous")
        cmap_colors.append(color)
        boundaries.append(lower)
        last_upper = upper
    boundaries.append(upper)
    cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm

def cmap_logarithmic(min_value, max_value, colors):#vole de panda power

    from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize, \
    LogNorm
    num_values = len(colors)
    if num_values < 2:
        raise UserWarning("Cannot create a logarithmic colormap less than 2 colors.")
    if min_value <= 0:
        raise UserWarning("The minimum value must be above 0.")
    if max_value <= min_value:
        raise UserWarning("The upper bound must be larger than the lower bound.")
    values = np.arange(num_values + 1)
    diff = (max_value - min_value) / (num_values - 1)
    values = (np.log(min_value + values * diff) - np.log(min_value)) \
             / (np.log(max_value) - np.log(min_value))
    cmap = LinearSegmentedColormap.from_list("name", list(zip(values, colors)))
    norm = LogNorm(min_value, max_value)
    return cmap, norm