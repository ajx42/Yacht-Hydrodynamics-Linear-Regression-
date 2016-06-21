# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:45:50 2016

@author: Aditya
"""

# Application of Linear Regression on Yacht Hydrodynamics Data Set
# Data was taken from UCI Machine Learning Repository
# source: https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics#

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

train = open("data.txt","r")
data = []
y = []
for i in enumerate(train):
    ls = list(map(float,train.readline().split()))   
    if len(ls)>0: 
        y.append(ls[6])
        data.append(ls[:6])

data = np.array(data)
y = np.array(y)

results = sm.OLS(y, data).fit()
print(results.summary())

train.close()

# Plots of y w.r.t various parameters

plt.plot([data[i][0] for i in range(len(data)) ],y,'ro')
plt.axis([-6,5,0,100])
plt.ylabel('Residuary resistance per unit weight of displacement')
plt.xlabel('Longitudinal position of the center of buoyancy')
plt.show()

plt.plot([data[i][1] for i in range(len(data)) ],y,'ro')
plt.axis([0.45,0.65,0,100])
plt.ylabel('Residuary resistance per unit weight of displacement')
plt.xlabel('Prismatic coefficient')
plt.show()

plt.plot([data[i][2] for i in range(len(data)) ],y,'ro')
plt.axis([4.5,5.5,0,100])
plt.ylabel('Residuary resistance per unit weight of displacement')
plt.xlabel('Length-displacement ratio')
plt.show()

plt.plot([data[i][3] for i in range(len(data)) ],y,'ro')
plt.axis([2,5.5,0,100])
plt.ylabel('Residuary resistance per unit weight of displacement')
plt.xlabel('Beam-draught ratio')
plt.show()

plt.plot([data[i][4] for i in range(len(data)) ],y,'ro')
plt.axis([2,4,0,100])
plt.ylabel('Residuary resistance per unit weight of displacement')
plt.xlabel('Length-beam ratio')
plt.show()

plt.plot([data[i][5] for i in range(len(data)) ],y,'ro')
plt.axis([0,0.5,0,100])
plt.ylabel('Residuary resistance per unit weight of displacement')
plt.xlabel('Froude number')
plt.show()
