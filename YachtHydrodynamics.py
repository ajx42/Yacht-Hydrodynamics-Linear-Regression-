# -*- coding: utf-8 -*-
"""
Created on Sat May 20 08:57:13 2017

@author: Aditya
"""


import numpy as np

'''
Read data from files
'''

fx = open('data.txt', 'r')
x = []
y = []
for line in fx:
    line = line.rsplit()
    x.append(line[:6])
    y.append(line[6])

x = np.array(x, dtype=float)
y = np.array(y, dtype=float)
fx.close()

m = len(y)


'''
Set learning rate and number of iterations
'''

alpha = 0.035
iters = 2000000

'''
Gradient Descent Method
'''

z = np.ones((x.shape[0], x.shape[1]+1))
z[:,:-1] = x
x = z
print(x.shape)

theta = np.zeros((x.shape[1],1))

for i in range(iters):
    predictions = np.dot(x, theta).flatten()
    k = predictions - y
    for j in range(7):
        theta[j,0] = theta[j,0] - alpha * (1./m) * np.sum(k*x[:,j]) 
    
'''
Result
'''
print(theta)

'''
[[   0.19384259]
 [  -6.42969928]
 [   4.23003433]
 [  -1.76453631]
 [  -4.51352054]
 [ 121.66755691]
 [ -19.23053779]]
'''


#import statsmodels.api as sm
#results = sm.OLS(y, x).fit()
#print(results.summary())    