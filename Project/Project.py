## PROJECT QUESTION#1 

import pandas as pd
import numpy as np
import math
from time import gmtime, strftime

#importing data
train_df = pd.read_csv('train.csv')
##test_df = pd.read_csv('test.csv')

train_df = train_df.iloc[:,:561]
##test_df = test_df.iloc[:,:561]

df = train_df
D = np.array(df)
##df = pd.merge(train_df, test_df, how = 'outer')


array = np.ones(561, dtype = int)
P = np.zeros((1,2))
n = len(D)

### count = -1              Use this when for loop has multiple k's
print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))



for k in [150]:         ## This for loop is used for multiple generations
    check = 0
    sqrt_k = math.pow(k,1/2)
    U = array * np.random.normal(0,1,(1,561))
    
    for i in range(k-1):
        u = array * np.random.normal(0,1,(1,561))
        U = np.append(U,u, axis = 0) 

    U_transpose = U.transpose()
    R = np.array(D.dot(U_transpose))        # New k dimensional subspace matrix
    
    for i in range(n-1):
        fvi = R[i,:]
        vi = D[i,:]
        for j in range(i+1,n):
            fvj = R[j,:]
            vj = D[j,:]      
            
            vij = np.linalg.norm(vi - vj)   ## Norm of pair difference from the dataset
            
            if (0.9) * (sqrt_k) * vij <= np.linalg.norm(fvi - fvj) <= (1.1) * (sqrt_k) * vij:
                check = check + 1
    
    prob = check/((7352*(7352-1))/2)
    
##  count = count + 1                  Use this when for loop has multiple k's
    P[0,0] = k                        # If there are multiple k's use P[k,0]
    P[0,1] = prob                     # If there are multiple k's use P[k,1]
    
print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))  

### BEST FIT ALONG THE SUBSPACE S ###

### Minimizing sum of (distance of points to best fit line)^2

### BY PYTHOGARIAN THEOREM

### (distance of point i to line)^2 = (length of row i)^2 - (length of projection)^2

sqr_sum = (np.linalg.norm(D)**2) - np.linalg.norm(R)**2



