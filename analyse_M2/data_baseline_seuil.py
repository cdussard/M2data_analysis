# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:00:21 2022

@author: claire.dussard
"""
from functions.preprocessData_eogRefait import *
essaisBaseline = ["5-1" for i in range(25)]
liste_rawPathBaseline = createListeCheminsSignaux(essaisBaseline,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

liste_rawPathBaseline = liste_rawPathBaseline[4:]
liste_data_baseline= read_raw_data(liste_rawPathBaseline)
liste_data_filtered = filter_data(liste_data_baseline,13,20,[])
for data in liste_data_filtered:
    data.pick_channels(["C3"])
events = mne.events_from_annotations(data)[0]


def computeMovingAverage(C3values,nvalues):
    arr_C3_movAverage = np.empty(nvalues, dtype=object) 
    for i in range(1,nvalues):
        print("n value"+str(i))
        point_1 = C3values[250*i :250*(i+1)].mean()
        point_2 = C3values[63+(250*i):63 + (250*(i+1))].mean()
        point_3 = C3values[125+(250*i) :125 +(250*(i+1))].mean()
        point_4 = C3values[188+(250*i) :188 +(250*(i+1)) ].mean()
        print(188+(250*i))
        pointMoyenne = (point_1+point_2+point_3 + point_4)/4
        arr_C3_movAverage[i] = pointMoyenne
    return arr_C3_movAverage
points = computeMovingAverage(np.mean(data._data,axis=0),240)
print(np.median(points[1:]))

print(min(min(liste_data_baseline[0].pick_channels( ["C3"])._data)))
print(max(max(liste_data_baseline[0].pick_channels( ["C3"])._data)))
#filter 13-20Hz
#le signal est mis au carré donc il est positif, ici il est negatif/positif (non rereference)
#on est quasi oblige de passer par un script vu que il faut filtrer 2 fichiers/ sujet
#dans 2 bandes differentes