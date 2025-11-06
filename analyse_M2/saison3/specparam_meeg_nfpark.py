# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:21:26 2024

@author: claire.dussard
"""



#==================== COMMENCER LE BORDEL================
liste_power_sujets_nf = [] #cf analysis_rest
for path in ls_paths:
    liste_power_sujets_nf.append(load_tfr_data_windows(path,""))

import specparam

from specparam import SpectralModel
# Import the FOOOF object
from fooof import FOOOF
fmin = 3
fmax = 84

freqs = np.arange(fmin,fmax+1, 1)

fmax_analyse = 40


r_2_fit_nf = []
fm_fit_nf = []
for i in range(len(ls_paths)):

    tfr = liste_power_sujets_nf[i]
    
    data = tfr._data
    
    reduced_data = np.median(data, axis=-1)
    
    elec = "Oz"
    one_elec = reduced_data[tfr.info.ch_names.index(elec)]
    
    fm = FOOOF(aperiodic_mode="knee",peak_width_limits=[2, 10],max_n_peaks=4,min_peak_height=0.1)
    fm.fit(freqs, one_elec, (fmin,fmax_analyse))
    
    
    fm.print_results()
    #fm.plot()
    
    r_2_fit_nf.append(fm.r_squared_)
    fm_fit_nf.append(fm)
    
    
    
min(r_2_fit_nf)

import matplotlib.pyplot as plt

# Exemple fictif : liste des modèles FOOOF (remplacez par vos modèles réels)
models_fooof_nf = fm_fit_nf # Liste de modèles FOOOF

# Configuration de la grille
n_cols = 4
n_rows = 3
n_plots = n_cols * n_rows  # Total 12 graphiques

# Créer une figure avec 12 sous-graphiques
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
axes = axes.flatten()  # Aplatir la grille pour un accès facile

# Remplir les graphiques
for i in range(n_plots):
    if i >= len(models_fooof_nf):  # Stopper si on dépasse le nombre de modèles
        break
    
    ax = axes[i]
    models_fooof_nf[i].plot(ax=ax,add_legend=False)  # Tracer le modèle FOOOF sur un axe spécifique
    ax.set_title(f"P{i + 1}")  # Ajouter un titre au graphique

# Supprimer les axes vides (si la liste de modèles a moins de 12 éléments)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Ajuster l'espacement et afficher la figure
plt.tight_layout()
plt.show()





#============= collect data ===========
import pandas as pd
import numpy as np

# Liste pour stocker les données
data_list_nf = []

# Parcourir tous les sujets
for i in range(len(ls_paths)):
    tfr = liste_power_sujets_nf[i]
    
    data = tfr._data
    reduced_data = np.median(data, axis=-1)
    
    for elec in tfr.info.ch_names:
        one_elec = reduced_data[tfr.info.ch_names.index(elec)]
        
        fm = FOOOF(aperiodic_mode="knee", peak_width_limits=[2, 10], max_n_peaks=4, min_peak_height=0.1)
        fm.fit(freqs, one_elec, (fmin, fmax_analyse))
        
        # Obtenir les résultats
        ap_params, peak_params, r_squared, fit_error, gauss_params = fm.get_results()
        
        # Calculate peak widths
        widths, width_heights, left_ips, right_ips = peak_widths(psd_med, peaks, rel_height=0.5)

        # Extract peak frequencies, prominence, and other properties
        peak_frequencies = freq_med[peaks]
        prominences = properties['prominences']
        
        # Extraire les informations
        for peak_param in peak_params:
            data_list_nf.append({
                'num_sujet': i + 1,  # Index du sujet (commence à 1 pour une numérotation humaine)
                'elec': elec,
                'peak_width': peak_param[2],   # Largeur des pics
                'peak_height': peak_param[1],  # Hauteur des pics
                'peak_freq': peak_param[0],    # Fréquence des pics
                'aperiodic_offset': ap_params[0],  # Offset aperiodique
                'knee': ap_params[1],              # Paramètre knee
                'exponent': ap_params[2],          # Exposant
                'r_squared': r_squared,            # R-squared
                'fit_error': fit_error ,
                'gauss_params': gauss_params,      # Paramètres Gaussien (s'il y en a)
                'groupe':'NFPARK'# Erreur de l'ajustement
        })

# Créer un DataFrame à partir de la liste
df = pd.DataFrame(data_list_nf)

# Afficher le DataFrame
print(df)

df.to_csv("../csv_files/meeg_baselineRest_NFPARK.csv")