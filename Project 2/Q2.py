import numpy as np
import pandas as pd

covariance = np.array(pd.read_csv("covariance.txt", sep=","))
P = np.array((pd.read_csv("P.txt", sep=",")))

eigenvalues, eigenvectors = np.linalg.eig(covariance)                           # Eigenvalues and Eigenvectors of the covariance matrix

first_value = sorted(eigenvalues, reverse=True)[0]                              # Highest eigenvalue of covariance matrix
second_value = sorted(eigenvalues, reverse=True)[1]                             # The second highest eigenvalue of covariance matrix

eigengap = first_value - second_value                                           # Eigengap calculation

maximal_vector = eigenvectors[:,np.argwhere(eigenvalues==first_value)[0,0]]     # Maximal Eigenvector of Covariance Matrix

perturbed_matrix = covariance + P                                               # Perturbed Matrix Calculation

eigenvalues_p, eigenvectors_p = np.linalg.eig(perturbed_matrix)                 # Eigenvalues and Eigenvectors of Perturbed Matrix

maximal_unique = eigenvectors_p[:,np.argwhere(eigenvalues_p == max(eigenvalues_p))[0,0]]

sorted_vectors = eigenvectors[:,np.argsort(-eigenvalues)]
sorted_values =  eigenvalues[np.argsort(-eigenvalues)]

def rank_k(evect, evalue, k):
    
    Z = np.zeros((100,100))
    for i in range(k):  
        Z += evalue[i]*np.dot(evect[:,i].reshape(100,1),evect[:,i].reshape(1,100))
    
    reconst = 0
    for j in range(k+1,100):
        reconst += evalue[j]
    return reconst, Z

const1 , z1 = rank_k(sorted_vectors,sorted_values,10)                           # Rank-k Projection for k = 10
const2 , z2 = rank_k(sorted_vectors,sorted_values,50)                           # Rank-k Projection for k = 50
const3 , z3 = rank_k(sorted_vectors,sorted_values,100)                          # Rank-k Projection for k = 100