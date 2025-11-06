# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:50:29 2023

@author: claire.dussard
"""

# Import the libraries
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt

# Create a random 2d matrix with values between 0 and 1
matrix = np.random.rand(10, 10)

# Plot the matrix with imshow
plt.imshow(matrix, cmap='viridis')
plt.colorbar()

# Create a contour of the values below 0.05
# Use the same data and extent as the imshow plot
# Set the levels to 0.05
# Set the colors to white
plt.contour(matrix, levels=[0.05], colors='white')

# Add some labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot with imshow and contour')

# Show the plot
plt.show()


matrix = np.array([[0.01, 0.02],
                   [0.03, 0.04]])
# Plot the matrix with imshow using nearest interpolation
plt.contour(matrix, levels=[0.03], colors='white')

plt.imshow(matrix, cmap='viridis', interpolation='none')
plt.colorbar()
plt.show()
# Create a contour of the values below 0.05 using nearest interpolation
# Use the same data and extent as the imshow plot
# Set the levels to 0.05
# Set the colors to white



matrix = np.random.rand(10, 10)

# Create a masked array where the values below 0.05 are masked
masked_matrix = np.ma.masked_where(matrix > 0.01, matrix)

# Plot the matrix with pcolor
# Use a grayscale colormap
# Set the edgecolors to white
# Set the linewidths to 2
plt.pcolor(matrix, cmap='viridis', linewidths=2)

plt.pcolor(masked_matrix, cmap='viridis',edgecolors="black", linewidths=2)

# Add some labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot with pcolor and masked array')

# Show the plot
plt.show()

#◘====== with my data

pvalue = 0.05
masked_global_pval = np.ma.masked_where(corr > pvalue,corr)
plt.pcolor(df_subset, cmap='RdBu_r', linewidths=2)

plt.pcolor(masked_global_pval, cmap='RdBu_r',edgecolors="black", linewidths=2, facecolors='none')
plt.colorbar()

#=============
pvalue = 0.05
masked_global_pval = np.ma.masked_where(corr > pvalue,corr)
pvalue = 0.001
masked_global_pval2 = np.ma.masked_where(corr > pvalue,corr)
plt.pcolor(df_subset, cmap='RdBu_r', linewidths=2)
plt.pcolor(masked_global_pval, cmap='RdBu_r',edgecolors="black", linewidths=1, facecolors='none')
plt.pcolor(masked_global_pval2, cmap='RdBu_r',edgecolors="green", linewidths=2, facecolors='none')

plt.colorbar()




#==== improve
vmax_scale = 28 
fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
freq_leg = np.arange(3,40,4)
freq_leg_str =[str(f) for f in freq_leg]
pos_freq = np.linspace(0.015,0.985,len(freq_leg))

vmin = 0
for i in range(len(pos_freq)):
    print(i)
    if i<3:
        pos_freq[i] = pos_freq[i]*(1-i*0.014)*vmax_scale
    elif i==3:
        pos_freq[i] = pos_freq[i]*(1-i*0.012)*vmax_scale
    elif i ==4:
        pos_freq[i] = pos_freq[i]*(1-i*0.008)*vmax_scale
    elif i==len(pos_freq)-1:
        print("last")
        pos_freq[i] = pos_freq[i]*(1-0.022)*vmax_scale
    elif i >=5:
        pos_freq[i] = pos_freq[i]*(1-i*0.004)*vmax_scale

plt.xticks(pos_freq,freq_leg_str)
# x8Hz = 0.1315
# x30Hz = 0.737
# col = "black"
# ls = "--"
# lw = 0.7
# axs.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
# axs.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs))*vmax_scale,elecs.iloc[::-1])

for elecPos in [0.107*vmax_scale,0.286*vmax_scale,0.428*vmax_scale,0.608*vmax_scale,0.75*vmax_scale,0.9293*vmax_scale]:
    axs.axhline(y=elecPos,color="dimgray",lw=0.25)
pvalue = 0.05
masked_global_pval = np.ma.masked_where(corr > pvalue,corr)
pvalue = 0.001
masked_global_pval2 = np.ma.masked_where(corr > pvalue,corr)
plt.pcolor(df_subset, cmap='RdBu_r', linewidths=2)
plt.pcolor(masked_global_pval, cmap='RdBu_r',edgecolors="black", linewidths=1, facecolors='none')
plt.pcolor(masked_global_pval2, cmap='RdBu_r',edgecolors="green", linewidths=2, facecolors='none')

plt.colorbar()