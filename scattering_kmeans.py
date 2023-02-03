import numpy as np
from sklearn.cluster import KMeans

#load saved scattering coefficients 
#X = np.loadtxt('/scratch/users/edarragh/comp_syn/CIFAR/scattering_coeffs_rgb.txt')

#initialize kmeans with kk clusters 
kk = 7
kmeans = KMeans(n_clusters=kk, n_init=100)

#perform fit
kmeans = kmeans.fit(X)


#predict labels
labels = kmeans.predict(X)


