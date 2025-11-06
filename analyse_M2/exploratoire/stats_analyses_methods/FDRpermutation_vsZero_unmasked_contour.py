# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:18:30 2024

@author: claire.dussard
"""


fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
img = axs.imshow(-main, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=0,vmax=2.1,label="agency")
contour = axs.contour(p_main,levels=[0.01], colors='black', linewidths=0.6, extent=[0, 1, 0, 1], corner_mask='legacy', origin='upper') 


pvalue = 0.05
p_pend_corr=  mne.stats.fdr_correction(p_pend)[1]
p_main_corr=  mne.stats.fdr_correction(p_main)[1]
p_mIll_corr=  mne.stats.fdr_correction(p_mIll)[1]

masked_p = np.ma.masked_where((p_pend_corr > pvalue) , pend)
masked_m = np.ma.masked_where((p_main_corr > pvalue) , main)
masked_mi = np.ma.masked_where((p_mIll_corr > pvalue) , mIll)


import matplotlib.pyplot as plt
elec_leg = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0]
gridspec_kw={'width_ratios': [1,1,1],
                           'height_ratios': [1],
                       'wspace': 0.05,#constrained_layout=True
                       'hspace': 0.05}
fig, axs = plt.subplots(1,3, sharey=True,sharex=True, figsize=(20, 7),gridspec_kw=gridspec_kw,constrained_layout=True)
vmin = 0.44#0.56
vmax = 2.13#2
img = axs[0].imshow(-pend, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="pendulum")
axs[0].text(0.12, 1.02, 'Virtual pendulum')

axs[1].imshow(-main, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax)
axs[1].text(0.12, 1.02, 'Virtual hand')
axs[2].imshow(-mIll, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax)
axs[2].text(0.12, 1.02, 'Virtual hand with vibrations')
fig.colorbar(img, location = 'right')
elecs = elec_leg 
#plt.subplots_adjust(wspace=0.2, hspace=0.05)
freq_leg = np.arange(3,84,4)
freq_leg_str =[str(f) for f in freq_leg]
plt.xticks(np.linspace(0,1,21),freq_leg_str)
x8Hz = 0.061
x30Hz = 0.3415
col = "black"
ls = "--"
lw = 0.7
for ax in axs.flat:
    ax.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
    ax.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for ax in axs.flat:
    for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
        ax.axhline(y=elecPos,color="dimgray",lw=0.25)

                
contour = axs[0].contour(p_pend_corr,levels=[0.05], colors='black', linewidths=0.6, extent=[0, 1, 0, 1], corner_mask='legacy', origin='upper') 
contour = axs[1].contour(p_main_corr,levels=[0.05], colors='black', linewidths=0.6, extent=[0, 1, 0, 1], corner_mask='legacy', origin='upper') 
contour = axs[2].contour(p_mIll_corr,levels=[0.05], colors='black', linewidths=0.6, extent=[0, 1, 0, 1], corner_mask='legacy', origin='upper') 

#plt.tight_layout(pad=0.04) 
plt.show()



