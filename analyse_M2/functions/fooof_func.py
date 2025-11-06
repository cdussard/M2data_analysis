# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:47:24 2024

@author: claire.dussard
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:42:33 2024

@author: claire.dussard
"""


def return_range_dispo_cond(epochs_jetes_cond):
    range_dispo = [i for i in range(1,11)]
    if len(epochs_jetes_cond)>0: 
        for item in epochs_jetes_cond:
            range_dispo.remove(item)
    return range_dispo



def return_sujet_data_cond(num_sujet,name_cond,power_cond_allTrials,freq_range,tmin,tmax,essais_dispo,report):
    df_cond = pd.DataFrame(columns=["num_sujet","num_essai","FB","freq","hauteur","largeur"])
    fg = plot_foofGroup_data(power_cond_allTrials,freq_range,tmin,tmax,True)
    for i in range(len(power_cond_allTrials)):
        fm = fg.get_fooof(ind=i, regenerate=True)
        if report:
            directory = "../pdf_files/"+str(num_sujet)+"/"+name_cond+"/"
            if not os.path.exists(directory): 
                os.makedirs(directory) 
            fm.save_report("report"+str(num_sujet)+name_cond+str(i), file_path=directory)
            #fm.report()
        params_peak = fm.get_params("peak_params")
        for ligne in params_peak:
            data_peak = {
                "num_sujet":num_sujet,
                "num_essai":essais_dispo[i],#modifie
                "FB":name_cond,
                "freq":round(ligne[0],1),
                "hauteur":round(ligne[1],2),
                "largeur":round(ligne[2],2),
                "Rsquared_fit":round(fm.get_params("r_squared"),2),
                "aperiodic_exp":round(fm.get_params("aperiodic_params")[1],2),
                "aperiodic_offset":round(fm.get_params("aperiodic_params")[0],2)
                }
            print(data_peak)
            df_cond = df_cond.append(data_peak,ignore_index=True)
    return df_cond
       
def return_sujet_data(num_sujet,freq_range,tmin,tmax,suffix,report):
    #read data
    print("loading data")
    power_pendule = load_tfr_data_windows(liste_rawPathPendule[num_sujet:num_sujet+1],suffix,True)[0]
    power_main = load_tfr_data_windows(liste_rawPathMain[num_sujet:num_sujet+1],suffix,True)[0]
    power_mainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion[num_sujet:num_sujet+1],suffix,True)[0]
    #create dataframe
    df_3cond = pd.DataFrame(columns=["num_sujet","num_essai","FB","freq","hauteur","largeur"])
    #fit FOOOF, return peak params
    real_num_sujet = real_sujets[num_sujet]
    df_pendule_sujet = return_sujet_data_cond(real_num_sujet,"pendule",power_pendule,freq_range,tmin,tmax,dispo_pendule[num_sujet],report)  
    df_main_sujet = return_sujet_data_cond(real_num_sujet,"main",power_main,freq_range,tmin,tmax,dispo_main[num_sujet],report)    
    df_mainIllusion_sujet = return_sujet_data_cond(real_num_sujet,"mainIllusion",power_mainIllusion,freq_range,tmin,tmax,dispo_mainIllusion[num_sujet],report)   
    df_3cond = pd.concat([df_pendule_sujet, df_main_sujet, df_mainIllusion_sujet], ignore_index=True)
    return df_3cond



#MI alone
def return_sujet_data_MIalone(num_sujet,freq_range,tmin,tmax,suffix,cond):
    #read data
    print("loading data")
    power_MI = load_tfr_data_windows(liste_rawPath_rawMIalone[num_sujet:num_sujet+1],suffix,True)[0]
    #power_MI.plot(picks="C3")
    df_cond = pd.DataFrame(columns=["num_sujet","FB","freq","hauteur","largeur"])
    #fit FOOOF, return peak params
    fg = plot_foof_data(power_MI,freq_range,tmin,tmax,True,"fixed")
    params_peak = fg.get_params("peak_params")
    for ligne in params_peak:
        data_peak = {
            "num_sujet":num_sujet,
            "FB":cond,
            "freq":round(ligne[0],1),
            "hauteur":round(ligne[1],2),
            "largeur":round(ligne[2],2),
            "Rsquared_fit":round(fg.get_params("r_squared"),2),
            "aperiodic_exp":round(fg.get_params("aperiodic_params")[1],2),
            "aperiodic_offset":round(fg.get_params("aperiodic_params")[0],2)
            }
        df_cond = df_cond.append(data_peak,ignore_index=True)
        print(df_cond)
    return df_cond



def plot_foof_data(power_object,freq_range,tmin,tmax,keepObject,aperiodic_mode):
    if keepObject:
        power_object = power_object.copy()
    power_object.pick_channels(["C3"])
    power_object.crop(tmin=tmin,tmax=tmax,fmin=min(freq_range),fmax=max(freq_range))
    print(power_object)
    freqs = power_object.freqs
    test = np.mean(power_object.data,axis=2)
    test = np.mean(test,axis=0)
    print(len(test)==len(freqs))
    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    mean_peak = 2*(freqs[1]-freqs[0])

    fm = FOOOF(peak_width_limits=(mean_peak,12),aperiodic_mode=aperiodic_mode)
    fm.fit(freqs, test, freq_range)
    fm.report(freqs, test, freq_range)
    return fm
    #fm.print_results()
    #fm.plot()
    
def plot_foofGroup_data(power_object,freq_range,tmin,tmax,keepObject):
    if keepObject:
        power_object = power_object.copy()
    power_object.pick_channels(["C3"])
    power_object.crop(tmin=tmin,tmax=tmax,fmin=min(freq_range),fmax=max(freq_range))
    freqs = power_object.freqs
    if min(freqs)==min(freq_range) and max(freqs)==max(freq_range):
        spectra = np.mean(power_object.data,axis=3)#mean over time avant : 3
        spectra = np.mean(spectra,axis=1)#mean over C3 electrode avant : 1 
        fg = FOOOFGroup(peak_width_limits=[2, 12], min_peak_height=0.1, max_n_peaks=4)
        fg.fit(freqs, spectra, freq_range)
        fg.report(freqs, spectra, freq_range)
        return fg
    else:
        print("ISSUE : input freq range not contained in original data")

