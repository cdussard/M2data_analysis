# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:28:40 2023

@author: claire.dussard
"""

EpochDataMain

#check events 
#pour main

def get_stats_trialDuration(liste_raw,numStimDebut):
    i = 0
    ls = []
    for signal in liste_raw:
        print(i)
        events = mne.events_from_annotations(signal)
        indicesDebutsMain = np.where(events[0][:,2] ==numStimDebut)[0]
        tpsDebutMain= events[0][indicesDebutsMain][:,0]
        indicesFinMain = np.where(events[0][:,2] ==6)[0]
        if len(indicesFinMain)==0:
            indicesFinMain = np.where(events[0][:,2] ==800)[0]#stim restauree depuis OpenVibe
        tpsFinMain= events[0][indicesFinMain][:,0]
        print(tpsFinMain)
        print(tpsDebutMain)
        diffTemps = tpsFinMain - tpsDebutMain
        ls.append(diffTemps)
        i += 1
    ls_flat  = np.concatenate(ls).ravel()
    
    mean = ls_flat.mean()
    sd = ls_flat.std()
    return ls,ls_flat,mean,sd


liste_raw = read_raw_data(liste_rawPathMain)
ls,ls_flat,mean,sd = get_stats_trialDuration(liste_raw,3)

ls_flat = array([25766, 25500, 25767, 25498, 25516, 25736, 25501, 25720, 25501,
       25501, 26533, 26516, 26221, 26517, 26517, 26251, 26485, 26283,
       26533, 26486, 26512, 26502, 26235, 26501, 26501, 26237, 26501,
       26282, 26501, 26485, 26532, 26484, 26235, 26486, 26501, 26234,
       26516, 26267, 26501, 26501, 26532, 26516, 26235, 26486, 26470,
       26250, 26501, 26282, 26501, 26469, 26516, 26517, 26266, 26516,
       26501, 26251, 26484, 26267, 26486, 26485, 26532, 26502, 26251,
       26486, 26502, 26251, 26486, 26267, 26502, 26485, 26532, 26501,
       26219, 26501, 26501, 26282, 26486, 26282, 26517, 26501, 26401,
       26401, 26401, 26401, 26401, 26401, 26401, 26401, 26401, 26401,
       26517, 26501, 26220, 26501, 26502, 26298, 26486, 26267, 26485,
       26486, 26547, 26485, 26219, 26486, 26487, 26298, 26485, 26251,
       26501, 26501, 26542, 26484, 26235, 26485, 26501, 26267, 26517,
       26298, 26501, 26485, 26531, 26501, 26235, 26502, 26500, 26266,
       26486, 26282, 26485, 26485, 26568, 26470, 26220, 26501, 26500,
       26282, 26516, 26282, 26484, 26486, 26236, 26486, 26268, 26485,
       26501, 26501, 26501, 26282, 26486, 26485, 26266, 26502, 26298,
       26484, 26501, 26507, 26502, 26250, 26501, 26501, 26251, 26502,
       26267, 26501, 26532, 26486, 26485, 26266, 26500, 26485, 26267,
       26516, 26266, 26501, 26517, 26501, 26485, 26283, 26500, 26485,
       26252, 26501, 26267, 26485, 26501, 26500, 26500, 26283, 26501,
       26502, 26251, 26485, 26251, 26486, 26501, 26501, 26501, 26282,
       26485, 26501, 26236, 26501, 26267, 26469, 26485, 26508, 26486,
       26282, 26516, 26516, 26252, 26486, 26267, 26484, 26486, 26407,
       26391, 26423, 26392, 26376, 26407, 26439, 26392, 26391, 26408])


liste_raw_p = read_raw_data(liste_rawPathPendule)
ls,ls_flat,mean,sd = get_stats_trialDuration(liste_raw_p,4)

ls_flat = np.array([26548, 26485, 26252, 26502, 26491, 26252, 26486, 26267, 26517,
       26502, 26549, 26501, 26267, 26502, 26517, 26236, 26486, 26267,
       26501, 26501, 26560, 26500, 26220, 26501, 26517, 26251, 26500,
       26267, 26501, 26501, 26531, 26516, 26251, 26501, 26485, 26251,
       26501, 26250, 26501, 26470, 26533, 26486, 26235, 26485, 26516,
       26250, 26501, 26267, 26500, 26486, 26516, 26486, 26204, 26500,
       26500, 26267, 26501, 26267, 26517, 26517, 26532, 26487, 26235,
       26517, 26502, 26268, 26501, 26282, 26471, 26486, 26532, 26500,
       26219, 26501, 26516, 26267, 26500, 26266, 26501, 26469, 26532,
       26485, 26251, 26501, 26486, 26282, 26501, 26266, 26502, 26485,
       26516, 26501, 26235, 26485, 26501, 26265, 26516, 26267, 26486,
       26469, 26273, 26470, 26251, 26486, 26517, 26267, 26501, 26299,
       26501, 26501, 26532, 26501, 26235, 26500, 26485, 26266, 26469,
       26282, 26485, 26486, 26502, 26501, 26251, 26500, 26517, 26266,
       26501, 26267, 26439, 26501, 26532, 26531, 26267, 26500, 26517,
       26251, 26501, 26282, 26501, 26485, 26501, 26469, 26220, 26487,
       26502, 26283, 26517, 26282, 26517, 26517, 26532, 26486, 26236,
       26534, 26502, 26266, 26517, 26250, 26501, 26501, 26547, 26501,
       26250, 26486, 26485, 26282, 26487, 26267, 26501, 26485, 26332,
       26516, 26235, 26485, 26501, 26267, 26470, 26251, 26501, 26501,
       26576, 26501, 26235, 26485, 26533, 26266, 26484, 26266, 26532,
       26485, 26516, 26516, 26282, 26485, 26501, 26266, 26500, 26234,
       26486, 26501, 26516, 26500, 26235, 26501, 26485, 26266, 26486,
       26282, 26501, 26485, 26516, 26502, 26251, 26500, 26470, 26283,
       26501, 26266, 26500, 26486, 26533, 26486, 26187, 26502, 26485,
       26267, 26501, 26283, 26470, 26500])

for signal in liste_raw:
    events = mne.events_from_annotations(signal)
    print(np.where(events[0][:,2] ==27)[0])
    
    
#entre le 3/4 et le premier 22
ls = []
i = 0
for signal in liste_raw[1:]:
    print(i)
    events = mne.events_from_annotations(signal)
    print(events)
    indicesDebutsMain = np.where(events[0][:,2] ==3)[0]
    tpsDebutMain= events[0][indicesDebutsMain][:,0]
    indicesFinMain = np.where(events[0][:,2] ==22)[0][0::16]#first 22 every 16 stim
    if len(indicesFinMain)==0:
        indicesFinMain = np.where(events[0][:,2] ==10002)[0][0::16]#stim restauree depuis OpenVibe
    tpsFinMain= events[0][indicesFinMain][:,0]
    #print(tpsDebutMain)
    print("len 22"+str(len(tpsFinMain)))
    diffTemps = tpsFinMain - tpsDebutMain
    print(diffTemps)
    ls.append(diffTemps)
    i += 1
    
ls_flat  = np.concatenate(ls).ravel()


#entre le dernier 22 et le 6
ls = []
i = 0
for signal in liste_raw[1:]:
    print(i)
    events = mne.events_from_annotations(signal)
    indicesDebutsMain = np.where(events[0][:,2] ==6)[0]
    tpsDebutMain= events[0][indicesDebutsMain][:,0]
    indicesFinMain = np.where(events[0][:,2] ==22)[0][15::16]#first 22 every 16 stim
    print(indicesFinMain)
    if len(indicesFinMain)==0:
        indicesFinMain = np.where(events[0][:,2] ==32771)[0][0::16]#stim restauree depuis OpenVibe
    elif len(indicesFinMain)==0:
        indicesFinMain = np.where(events[0][:,2] ==10002)[0][0::16]#stim restauree depuis OpenVibe
    tpsFinMain= events[0][indicesFinMain][:,0]
    diffTemps = tpsDebutMain - tpsFinMain 
    print(diffTemps)
    ls.append(diffTemps)
    i += 1
    
ls_flat  = np.concatenate(ls).ravel()


#entre les differents 22
ls = []
i = 0
for signal in liste_raw[1:]:
    print(i)
    events = mne.events_from_annotations(signal)
    indicesFinMain = np.where(events[0][:,2] ==22)[0]#first 22 every 16 stim
    print(indicesFinMain)
    if len(indicesFinMain)==0:
        indicesFinMain = np.where(events[0][:,2] ==32771)[0][0::16]#stim restauree depuis OpenVibe
    elif len(indicesFinMain)==0:
        indicesFinMain = np.where(events[0][:,2] ==10002)[0][0::16]#stim restauree depuis OpenVibe
    tpsFinMain= events[0][indicesFinMain][:,0]
    diffTemps =     [x - tpsFinMain[i - 1] for i, x in enumerate(tpsFinMain)][1:][0:15]
    print(diffTemps)
    ls.append(diffTemps)
    i += 1
    
ls_flat  = np.concatenate(ls).ravel()


#check unit 

data_8_30 = av_power_pendule.copy().crop(fmin=8,fmax=30,tmin=2.5,tmax=25.5)._data
dat = np.concatenate(data_8_30).ravel()
min(dat)
max(dat)
