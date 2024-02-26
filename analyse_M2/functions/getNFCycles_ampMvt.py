# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:45:01 2023

@author: claire.dussard
"""

import pandas as pd
import numpy as np

data = pd.read_excel("../../../../csv_files/H3_GetNFCyclesAmpMvt/dataVisualFeedback_8_30_add.xlsx")

jetes_main = [
    [],[],[],[3],[2,4,5,6,7],[7],[],[6],[9,10],[8],[6],
    [1,6,8],[1,10],[9,10],[6,7,8,9,10],[3,6],[3,6,7],[4,10],[],[1,6],[],[9],[]
    ]

jetes_pendule = [
    [],[],[],[5],[1,7,10],[],[],[3,5,8,10],[],[5,10],[],
    [5,6],[4],[6,9],[],[9],[3,8,9],[],[],[1,6],[6],[3,9],[6,8]
    ]

jetes_mainIllusion = [
    [6],[1,3,6],[1,2],[],[5,6,8,9,10],[],[],[1,6,7,8],[6,7,8,9,10],[4,10],[1],
    [],[1,8,10],[10],[6,9],[9],[4,8,9],[4,8],[],[1,6],[],[1],[]
    ]

data_main = data[data["FB_int"]==1]


#remove suj 1 et suj 4

