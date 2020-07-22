## PROJECT QUESTION#3

import pandas as pd
import numpy as np
import math
from time import gmtime, strftime

print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

train_df = pd.read_csv('train.csv')

train_df = train_df.iloc[:,:561]

D = np.array(train_df)

svd_d = np.linalg.svd(D)    ##Full SVD for Data D 
eig_values_d = svd_d[1]     ##EigenValues Vector

trace_matrix = np.ones((561,561))*eig_values_d
eig_values_trace_sqrt = np.linalg.svd(trace_matrix**(1/2))[1]

qnt = 100   #qnt for simulations for probability estimate

check_p1 = 0
check_p2 = 0
for x in range(qnt):            
       
    gd = np.random.normal(0, eig_values_d[0], (1,561))          ## initilialization and first row generation of new data
    for i in range(1,561):
        row = np.random.normal(0,eig_values_d[i],(1,561))       ## every row is N(0,Var(corresponding eigenvalue))
        gd = np.append(gd, row, axis = 0)
        
    eig_values_gd  = np.linalg.svd(gd)[1]  ## EigenValues of New Generated Data
    
    ## IF STATEMENTS FOR PROBABLITY ESTIMATION
    if eig_values_gd[0] / math.sqrt(len(D)) >= 1.05 * max(eig_values_trace_sqrt) + math.sqrt(np.trace(trace_matrix)/len(trace_matrix)):
        check_p1 = check_p1 + 1
        
    if eig_values_gd[560] / math.sqrt(len(D)) >= 0.95 * min(eig_values_trace_sqrt) - math.sqrt(np.trace(trace_matrix)/len(trace_matrix)):
        check_p2 = check_p2 + 1
        
    
prob1 = check_p1 / qnt    # PROBABABILITY ESTIMATION FOR THE FIRST STATEMENT
prob2 = check_p2 / qnt    # PROBABABILITY ESTIMATION FOR THE FIRST STATEMENT
    
print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))        
    
    
