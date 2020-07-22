## PROJECT QUESTION#2

import pandas as pd
import numpy as np
import math
from time import gmtime, strftime

train_df = pd.read_csv('train.csv')

train_df = train_df.iloc[:,:561]

df = train_df

D = np.array(df)


singular_v = np.linalg.svd(D)[2]    ## singular vectors of full SVD of the Data {index 0: U, index 1: EigenValue Matrix E, index 2: V}

fits = np.ones((561,2)) # initialization

sqr_points = np.linalg.norm(D)**2   ## sum of data's row distances squared 


## for loop for finding fit values for every K-SVD Subspaces k = 1,2,...,d

## fit objective:
## <minimizing> (distance of point i to line)^2 = (length of row i)^2 - (length of projection)^2
for k in range(561):
    
    sqr_dis = sqr_points - np.linalg.norm(D.dot(np.transpose(singular_v[0:k+1,:])))**2
    
    fits[k,0] = k
    fits[k,1] = sqr_dis
    
## K-SVD Fits / Full-SVD Fit    
ratio = ( fits[560,1] /fits[:,1] /fits[560,1] )

