# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:29:55 2024

@author: claire.dussard
"""

import mne
import pandas as pd
easycap_montage = mne.channels.make_standard_montage("easycap-M1")
EpochDataMain = load_data_postICA_postdropBad_windows(liste_rawPathMain[0:1],"",True)[0]
EpochDataMain.drop_channels(["HEOG","VEOG","TP9","TP10","FT9","FT10"])
EpochDataMain.set_montage('easycap-M1')       

EpochDataMain.plot_sensors(show_names=True)
plt.show()
fig = easycap_montage.plot(kind="3d", show=False)  # 3D
fig = fig.gca().view_init(azim=70, elev=15) 

EpochDataMain.info["dig"].to_dataframe()
EpochDataMain.info.ch_names

len(EpochDataMain.info.ch_names)
len(EpochDataMain.info["dig"])

df_locations = pd.DataFrame(columns=["elec","X","Y","Z"])

for (line,elec) in zip(EpochDataMain.info["dig"][3:],EpochDataMain.info.ch_names):
    df_line = {
        "elec":elec,
        "X":line["r"][0],
        "Y":line["r"][1],
        "Z":line["r"][2],
        
        }
    print(df_line)
    df_locations= df_locations.append(df_line,ignore_index=True)
    
df_locations.to_csv("cartesian_coordinates.csv")