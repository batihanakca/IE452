from scipy.io import loadmat     # For loading the .mat data
import numpy as np               # For vector/matrix manipulation
import sklearn.decomposition     # For running PCA
import matplotlib.pyplot as plt  # For plots

data = loadmat('numbers.mat')                                                   # Loading the data
digits = np.array (data['digits'])
labels = np.array (data['labels'])

pca = sklearn.decomposition.PCA(len(digits[0]))                                 # Full PCA for eigenvalues and the components
pca.fit(digits)

digits_pca_full = pca.transform(digits)

eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
sample_mean = pca.mean_
covariance_matrix = pca.get_covariance()

first_component = eigenvectors[0,:]
second_component = eigenvectors[1,:]
third_component = eigenvectors[2,:]


x = np.linspace(1, len(eigenvalues), len(eigenvalues))
y = pca.explained_variance_ratio_
for i in range(399):
    y[i+1] = y[i] + y[i+1]
      
    
fig0, ax = plt.subplots()                                                       # Plotting cumulative sum of explained variance by top i eigenvalues.         
ax.plot(x, y)                                   
ax.set_xlabel("Number of chosen components")
ax.set_ylabel("Cumulative sum of explained variance")
ax.set_title("Explained Variance Cumulative")
fig0.savefig("EigenvaluesCumulative.png",dpi = 1200)

                          
fig1, ax = plt.subplots()                                                       # Plotting for the eigenvalues in descending order
ax.plot(x, eigenvalues, label ="Eigenvalues")
ax.set_xlabel("Order of the eigenvalues")
ax.set_ylabel("Eigenvalue")
ax.set_title("Eigenvalues in Descending Order")
fig1.savefig("Eigenvalues.png",dpi = 1200)


    
pca = sklearn.decomposition.PCA(2)                                              # PCA for the first and the second components.
pca.fit(digits)
digits_reduced = pca.transform(digits)

fig2, ax = plt.subplots()                                                       # Plotting the projected data into the first two components.
for i in range(10):
    population = digits_reduced[(labels == i).reshape(5000,)]
    ax.scatter(x = digits_reduced[(labels == i).reshape(5000,)][:,0], 
               y = digits_reduced[(labels == i).reshape(5000,)][:,1], 
               label = str(i))
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
ax.legend(fancybox=True, framealpha=0.4)
fig2.savefig("The first and the second component.png",dpi = 1200)

digits_reduced_2 = np.append(digits_pca_full[:,0].reshape(5000,1),digits_pca_full[:,2].reshape(5000,1),axis=1)  # Projection into the first and the third component

fig3, ax = plt.subplots()
for i in range(10):                                                             # Plotting the projected data into the first and the third components.
    population = digits_reduced_2[(labels == i).reshape(5000,)]
    ax.scatter(x = digits_reduced_2[(labels == i).reshape(5000,)][:,0], 
               y = digits_reduced_2[(labels == i).reshape(5000,)][:,1], 
               label = str(i))
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Third Principal Component")
ax.legend(fancybox=True, framealpha=0.4)
fig3.savefig("The first and the third component.png",dpi = 1200)

plt.figure(figsize=(9, 3))                                                      # Images of the bases/components.
plt.subplot(131)
plt.imshow(np.transpose(first_component.reshape(20,20)),cmap='gray')
plt.axis("off")
plt.title("First P. Component")
plt.subplot(132)
plt.imshow(np.transpose(second_component.reshape(20,20)),cmap='gray')
plt.axis("off")
plt.title("Second P. Component")
plt.subplot(133)
plt.imshow(np.transpose(third_component.reshape(20,20)),cmap='gray')
plt.axis("off")
plt.title("Third P. Component")
plt.savefig("three_components.png",dpi = 1200)

plt.figure()                                                                    # Image of the sample mean.
plt.imshow(np.transpose(sample_mean.reshape(20,20)),cmap='gray')
plt.title("Sample Mean Image")
plt.axis("off")
plt.savefig("sample_mean.png",dpi = 1200)


v = np.linspace(10,200,20)
projected = list(np.linspace(10,200,20))
k = 0
for i in v:                                                                     # Projection of 20 different subspaces    
    pca = sklearn.decomposition.PCA(int(i))                                     # list 'projected' contains reduced dimensions
    pca.fit(digits)                                                             # {10, 20, 30, ......, 200}
    projected[k] = pca.transform(digits)
    k += 1
