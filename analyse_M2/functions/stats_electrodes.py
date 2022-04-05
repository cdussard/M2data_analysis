#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:24:23 2021

@author: claire.dussard
"""
#stats for each subject
def return_rawStats(liste_rawPath):
    listeRaw = read_raw_data(liste_rawPath)
    #amplitude moyenne des raw_files
    column_names = listeRaw[0].ch_names
    df_stats_amp = pd.DataFrame(columns=column_names)
    i = 0
    for raw in listeRaw:
        mean_raw_data = raw.get_data()
        mean_amp_elec = mean_raw_data.mean(axis=1) * 1e6
        df_stats_amp.loc[i] = mean_amp_elec
        i += 1
    df_stats_amp.columns = column_names
    return listeRaw,df_stats_amp


event_idMain={'Essai_main':3}  
event_idPendule={'Essai_pendule':4} 
event_idMainIllusion = {'Essai_mainIllusion':3}
event_idMainTactile = {'Essai_mainTactile':3}

def return_epochStats(event_id,liste_rawPath):
#mean amplitude epochs
    listeEpochs = return_epochs(liste_rawPath,0.1,90,[50,100],event_id)
    print(listeEpochs)
    i = 0 #num_sujet
    column_names = listeEpochs[0].ch_names
    df_stats_amp = pd.DataFrame(columns=column_names)
    for epochs in listeEpochs:
        mean_epoch_data = listeEpochs[i].get_data().mean(axis=0)
        mean_amp_elec = mean_epoch_data.mean(axis=1) * 1e6
        df_stats_amp.loc[i]=mean_amp_elec
        i += 1 
    return listeEpochs,df_stats_amp

liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)
liste_rawPathMainTactile = createListeCheminsSignaux(essaisMainTactile,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)

#stats main seule
listeEpochsMain,df_stats_ampMain = return_epochStats(event_idMain,liste_rawPathMain)
df_stats_ampMain.to_csv("csv_files/stats_epochs/stats_epochs_sujets_main.csv")#save source file
listeRawMain, statsMain = return_rawStats(liste_rawPathMain)

#resume par electrode
df_stats_ampMain_epochs_meanSujets = abs(df_stats_ampMain).mean(axis=0).sort_values(ascending=False)#des fois neg des fois pos : mean sur l'abs plutot que abs sur mean
df_stats_ampMain_raw_meanSujets = abs(statsMain).mean(axis=0).sort_values(ascending=False)
#mediane
df_stats_ampMain_epochs_medianSujets = abs(df_stats_ampMain).median(axis=0).sort_values(ascending=False)
df_stats_ampMain_raw_medianSujets = abs(statsMain).median(axis=0).sort_values(ascending=False)


print(df_stats_ampMain_epochs_meanSujets)
print(df_stats_ampMain_raw_meanSujets)

#stats pendule
listeEpochsPendule,df_stats_ampPendule = return_epochStats(event_idPendule,liste_rawPathPendule)
df_stats_ampPendule.to_csv("../csv_files/stats_epochs/stats_epochs_sujets_pendule.csv")#save source file
liste_rawPendule,statsPendule = return_rawStats(liste_rawPathPendule)
#resume par electrode
df_stats_ampPendule_epochs_meanSujets = abs(df_stats_ampPendule).mean(axis=0).sort_values(ascending=False)#des fois neg des fois pos : mean sur l'abs plutot que abs sur mean
df_stats_ampPendule_raw_meanSujets = abs(statsPendule).mean(axis=0).sort_values(ascending=False)
#mediane
df_stats_ampPendule_epochs_medianSujets = abs(df_stats_ampPendule).median(axis=0).sort_values(ascending=False)
df_stats_ampPendule_raw_medianSujets = abs(statsPendule).median(axis=0).sort_values(ascending=False)

print(df_stats_ampPendule_epochs_meanSujets)
print(df_stats_ampPendule_raw_meanSujets)

#stats main illusion
listeEpochsMainIllusion,df_stats_ampMainIllusion = return_epochStats(event_idMainIllusion,liste_rawPathMainIllusion)
df_stats_ampMainIllusion.to_csv("../csv_files/stats_epochs/df_stats_ampMainIllusion.csv")
listeRawMainIllusion, statsMainIllusion = return_rawStats(liste_rawPathMainIllusion)
#resume par electrode
df_stats_ampMainIllusion_epochs_meanSujets = abs(df_stats_ampMainIllusion).mean(axis=0).sort_values(ascending=False)#des fois neg des fois pos : mean sur l'abs plutot que abs sur mean
df_stats_ampMainIllusion_raw_meanSujets = abs(statsMainIllusion).mean(axis=0).sort_values(ascending=False)
#mediane
df_stats_ampMainIllusion_epochs_medianSujets = abs(df_stats_ampMainIllusion).median(axis=0).sort_values(ascending=False)
df_stats_ampMainIllusion_raw_medianSujets = abs(statsMainIllusion).median(axis=0).sort_values(ascending=False)

print(df_stats_ampMainIllusion_epochs_meanSujets)
print(df_stats_ampMainIllusion_raw_meanSujets)
print(df_stats_ampMainIllusion_epochs_medianSujets)
print(df_stats_ampMainIllusion_raw_medianSujets)

#stats mainTactile
listeEpochsMainTactile,df_stats_ampMainTactile = return_epochStats(event_idMainTactile,liste_rawPathMainTactile)
listeRawMainTactile, statsMainTactile = return_rawStats(liste_rawPathMainTactile)
df_stats_ampMainTactile.to_csv("df_stats_ampMainTactile.csv")
#resume par electrode
df_stats_ampMainTactile_epochs_meanSujets = abs(df_stats_ampMainTactile).mean(axis=0).sort_values(ascending=False)#des fois neg des fois pos : mean sur l'abs plutot que abs sur mean
df_stats_ampMainTactile_raw_meanSujets = abs(statsMainTactile).mean(axis=0).sort_values(ascending=False)
#mediane
df_stats_ampMainTactile_epochs_medianSujets = abs(df_stats_ampMainTactile).median(axis=0).sort_values(ascending=False)
df_stats_ampMainTactile_raw_medianSujets = abs(statsMainTactile).median(axis=0).sort_values(ascending=False)

print(df_stats_ampMainTactile_epochs_meanSujets)
print(df_stats_ampMainTactile_raw_meanSujets)
print(df_stats_ampMainTactile_epochs_medianSujets)
print(df_stats_ampMainTactile_raw_medianSujets)

#il faudrait utiliser ces tableurs pour exclure les elecs ? si on exclut des elecs differentes selon la condition pb ?
#mais si on parcourt tout sujet par sujet, ca revient un peu au mark bad via reject amplitude 
#(mais diff = ici base sur moyenne/mediane vs reject = des que amplitude > ca saute)

def find_bad_electrodes(df_stats):
    liste_bad_electrodes_values = []
    listes_bad_electrodes = []
    df_stats_hors_eog = df_stats.drop(["VEOG","HEOG","EMG"],axis=1)
    channels = df_stats_hors_eog.columns
    for i in range(len(df_stats_hors_eog)):
        print("\nsujet "+str(allSujetsDispo[i]))
        sujet = abs(df_stats_hors_eog.loc[i]) #arg_max = sujet.argmax(axis=0)  #print(channels[arg_max])
        noisy_electrodes = sujet[sujet>2*sujet.mean(axis=0)]
        for index,electrode in noisy_electrodes.items(): 
            print(str(index)+": "+str(round(electrode/sujet.mean(axis=0),1))+" fois la moyenne des électrodes autour")
        liste_bad_electrodes_values.append(noisy_electrodes)
        listes_bad_electrodes.append(noisy_electrodes.index.to_list())
        #print(noisy_electrodes)
    return liste_bad_electrodes_values,listes_bad_electrodes

liste_bad_electrodes_values_mainTactile, listes_bad_electrodes_mainTactile = find_bad_electrodes(df_stats_ampMainTactile)
liste_bad_electrodes_values_main, listes_bad_electrodes_main = find_bad_electrodes(df_stats_ampMain)
liste_bad_electrodes_values_pendule, listes_bad_electrodes_pendule = find_bad_electrodes(df_stats_ampPendule)
liste_bad_electrodes_values_mainIllusion, listes_bad_electrodes_mainIllusion = find_bad_electrodes(df_stats_ampMainIllusion)

# import sklearn
# from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9','CP5','CP1','CP2','CP6','TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'])
# #listes_bad_electrodes_mainIllusion)
# yo = encoder.fit(listes_bad_electrodes_mainIllusion)
# def one_hot_encode():
#     return matrix_conditions
