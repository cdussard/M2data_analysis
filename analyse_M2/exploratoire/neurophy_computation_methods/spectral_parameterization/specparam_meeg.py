# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:22:19 2024

@author: claire.dussard
"""

import os 
import seaborn as sns
import pathlib
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *
import numpy as np 

#create liste of file paths
essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData() 

nom_essai = "1-2"#on prend seulement le premier de 2 min pour l'instant faire simple

allSujetsDispo_rest = allSujetsDispo#S014 pas de marqueurs sur EEG, a voir + tard

allSujetsDispo_rest.append(4)

allSujetsDispo.pop()
        

#on commence au 3 (reste pbs noms)
liste_rawPath_rawRest = []
for num_sujet in allSujetsDispo_rest:
    print("sujet n° "+str(num_sujet))
    nom_sujet = listeNumSujetsFinale[num_sujet]
    if num_sujet in SujetsPbNomFichiers:
        if num_sujet>0:
            date_sujet = '19-04-2021'
        else:
            date_sujet = '15-04-2021'
    else:
        date_sujet = dates[num_sujet]
    date_nom_fichier = date_sujet[-4:]+"-"+date_sujet[3:5]+"-"+date_sujet[0:2]+"_"
    dateSession = listeDatesFinale[num_sujet]
    sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
    sample_data_dir = pathlib.Path(sample_data_loc)
    raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + nom_essai+".vhdr")#il faudrait recup les epochs et les grouper ?
    liste_rawPath_rawRest.append(raw_path_sample)

print(liste_rawPath_rawRest)


#pour se placer dans les donnees lustre
os.chdir("../../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)



        
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


liste_power_sujets = load_tfr_data_windows(liste_rawPath_rawRest,"",True)


#==================== COMMENCER LE BORDEL================


import specparam

from specparam import SpectralModel
# Import the FOOOF object
from fooof import FOOOF
fmin = 3
fmax = 84

freqs = np.arange(fmin,fmax+1, 1)

fmax_analyse = 40



r_2_fit = []
fm_fit = []
for i in range(len(liste_rawPath_rawRest)):

    tfr = liste_power_sujets[i]
    
    data = tfr._data
    
    reduced_data = np.median(data, axis=-1)
    
    elec = "C3"
    one_elec = reduced_data[tfr.info.ch_names.index(elec)]
    
    fm = FOOOF(aperiodic_mode="knee",peak_width_limits=[2, 10],max_n_peaks=4,min_peak_height=0.1)
    fm.fit(freqs, one_elec, (fmin,fmax_analyse))
    
    
    #fm.print_results()
    #fm.plot()
    
    r_2_fit.append(fm.r_squared_)
    fm_fit.append(fm)
    
    
    



#===========
import matplotlib.pyplot as plt

# Supposons que vous avez 23 modèles FOOOF stockés dans `models_fooof`
# Chaque modèle correspond à un `fm` avec une méthode `.plot()`

# Exemple fictif : liste des modèles FOOOF
models_fooof = fm_fit  # Remplacez par vos modèles réels

# Nombre de plots par figure
n_cols = 4
n_rows = 3
n_plots_per_fig = n_cols * n_rows

# Itérer sur les figures
for fig_num in range(2):  # 2 figures
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()  # Aplatir pour accès facile

    # Plots dans la figure courante
    for i in range(n_plots_per_fig):
        plot_idx = fig_num * n_plots_per_fig + i 
        if plot_idx >= len(models_fooof):  # Si dépasse la liste, arrêter
            break
        
        ax = axes[i]
        models_fooof[plot_idx].plot(ax=ax,add_legend=False)  # Tracer le modèle FOOOF sur un axe spécifique
        ax.set_title(f"HC {plot_idx + 1}")  # Ajouter un titre à chaque graphique

    # Supprimer les axes vides (si moins de 12 dans la dernière figure)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()






#============= collect data ===========
import pandas as pd
import numpy as np

# Liste pour stocker les données
data_list = []

# Parcourir tous les sujets
for i in range(len(liste_rawPath_rawRest)):
    tfr = liste_power_sujets[i]
    
    data = tfr._data
    reduced_data = np.median(data, axis=-1)
    
    for elec in tfr.info.ch_names:
        one_elec = reduced_data[tfr.info.ch_names.index(elec)]
        
        fm = FOOOF(aperiodic_mode="knee", peak_width_limits=[2, 10], max_n_peaks=4, min_peak_height=0.1)
        fm.fit(freqs, one_elec, (fmin, fmax_analyse))
        
        # Obtenir les résultats
        ap_params, peak_params, r_squared, fit_error, gauss_params = fm.get_results()
        
        # Extraire les informations
        for peak_param in peak_params:
            data_list.append({
                'num_sujet': i + 1,  # Index du sujet (commence à 1 pour une numérotation humaine)
                'elec': elec,
                'peak_width': peak_param[2],   # Largeur des pics
                'peak_height': peak_param[1],  # Hauteur des pics
                'peak_freq': peak_param[0],    # Fréquence des pics
                'aperiodic_offset': ap_params[0],  # Offset aperiodique
                'knee': ap_params[1],              # Paramètre knee
                'exponent': ap_params[2],          # Exposant
                'r_squared': r_squared,            # R-squared
                'fit_error': fit_error ,# Erreur de l'ajustement
                'gauss_params': gauss_params,      # Paramètres Gaussien (s'il y en a)
                'groupe':'BETAPARK'
        })

# Créer un DataFrame à partir de la liste
df = pd.DataFrame(data_list)

# Afficher le DataFrame
print(df)

df.to_csv("../csv_files/meeg_baselineRest_BETAPARK.csv")





