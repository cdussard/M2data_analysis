# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:16:23 2022

@author: claire.dussard
"""
data_pendule_8_30 = np.mean(av_power_pendule.data[11][5:27],axis=0)
data_main_8_30 = np.mean(av_power_main.data[11][5:27],axis=0)

plt.plot(av_power_pendule.times,data_pendule_8_30,label='pendule')
plt.plot(av_power_main.times,data_main_8_30,label='main')
plt.legend()

raw_signal.plot(block=True)

#rajouter shade avec SEM

liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule,"",True)
liste_tfrMain = load_tfr_data_windows(liste_rawPathMain,"",True)


dureePreBaseline = 3 #3
dureePreBaseline = - dureePreBaseline
dureeBaseline = 2.0 #2.0
valeurPostBaseline = dureePreBaseline + dureeBaseline

baseline = (dureePreBaseline, valeurPostBaseline)
for tfr_p,tfr_m in zip(liste_tfrPendule,liste_tfrMain):
    tfr_p.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_m.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    

data_suj_m = np.zeros((23,9000))
data_suj_p = np.zeros((23,9000))
for n_suj in range(len(liste_tfrPendule)):
  #  for timePoint in range(len(av_power_main.times)):
    data_suj_p[n_suj] =  np.mean(liste_tfrPendule[n_suj].data[11][5:27],axis=0)
    data_suj_m[n_suj] =  np.mean(liste_tfrMain[n_suj].data[11][5:27],axis=0)
        
        
#get std
std_p = []
std_m = []
for i in range(9000):
    std_p.append(np.std(data_suj_p[:,i]))
    std_m.append(np.std(data_suj_m[:,i]))
            
fig, ax = plt.subplots()
ax.fill_between(av_power_pendule.times,data_pendule_8_30-std_p,data_pendule_8_30+std_p,label='pendule',alpha=0.3)
ax.fill_between(av_power_main.times,data_main_8_30-std_m,data_main_8_30+std_m,label='main',alpha=0.3)
ax.plot(av_power_pendule.times,data_pendule_8_30,label='pendule_mean')
ax.plot(av_power_main.times,data_main_8_30,label='main_mean')
plt.legend()
raw_signal.plot(block=True)

def computeMovingAverage(C3values,nvalues):
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

yo = computeMovingAverage(data_pendule_8_30,28) 

val_pend = [val for val in yo if val is not None]
plt.plot(range(-3,16),val_pend)
raw_signal.plot(block=True)

data_suj_m = np.zeros((23,19))
data_suj_p = np.zeros((23,19))
for n_suj in range(len(liste_tfrPendule)):
    data_pendule_8_30 = np.mean(liste_tfrPendule[n_suj].data[11][5:27],axis=0)
    data_pend_i = computeMovingAverage(data_pendule_8_30,28) 
    data_pend_i = [val for val in data_pend_i if val is not None]
    data_main_8_30 = np.mean(liste_tfrMain[n_suj].data[11][5:27],axis=0)
    data_main_i = computeMovingAverage(data_main_8_30,28) 
    data_main_i = [val for val in data_main_i if val is not None]
    data_suj_p[n_suj] =  data_pend_i
    data_suj_m[n_suj] =  data_main_i
    print(data_pend_i)
    print(data_main_i)
    
#get std
std_p = []
std_m = []
for i in range(19):
    std_p.append(np.std(data_suj_p[:,i]))
    std_m.append(np.std(data_suj_m[:,i]))
    
    
data_av_pend = np.mean(av_power_pendule.data[11][5:27],axis=0)
p = computeMovingAverage(data_av_pend,28) 
p = [val for val in p if val is not None]  
data_av_main = np.mean(av_power_main.data[11][5:27],axis=0)  
m = computeMovingAverage(data_av_main,28) 
m = [val for val in m if val is not None]  

p = np.array(p)
m = np.array(m)

fig, ax = plt.subplots()
ax.fill_between(range(-3,16),p-std_p,p+std_p,label='pendule',alpha=0.3)
ax.fill_between(range(-3,16),m-std_m,m+std_m,label='main',alpha=0.3)
ax.plot(range(-3,16),p,label='pendule_mean')
ax.plot(range(-3,16),m,label='main_mean')
plt.legend()
raw_signal.plot(block=True)