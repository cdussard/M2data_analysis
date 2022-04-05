#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 19:03:47 2022

@author: claire.dussard
"""

liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)

nbSujets = 24
SujetsDejaTraites = 0
rawPath_main_sujets = liste_rawPathMain[SujetsDejaTraites:SujetsDejaTraites+nbSujets]

liste_ICA_main = load_ICA_files(rawPath_main_sujets)
#ica0 = liste_ICA_main[0]

#L'INFO de quel composant est soustrait est contenu dans exclude
#load toutes les ica 
#print les exclude et les noter dans un tableur
i = 0
for ica in liste_ICA_main:
    print("sujet  "+str(i))
    print(ica.exclude)
    i = i +1
 
    
#PENDULE 
rawPath_pendule_sujets = liste_rawPathPendule[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
liste_ICA_pendule = load_ICA_files(rawPath_pendule_sujets)

i = 0
for ica in liste_ICA_pendule:
    print("sujet  "+str(i))
    print(ica.exclude)
    i = i +1
    
#amelioration : faire un script qui genere un tableur plutot que copier coller les composantes

#Main VIBRATIONS
rawPath_mainIllusion_sujets = liste_rawPathMainIllusion[SujetsDejaTraites:SujetsDejaTraites+nbSujets]

liste_ICA_mainIllusion = load_ICA_files(rawPath_mainIllusion_sujets)
#ica0 = liste_ICA_main[0]

i = 0
for ica in liste_ICA_mainIllusion:
    print("sujet  "+str(i))
    print(ica.exclude)
    i = i +1