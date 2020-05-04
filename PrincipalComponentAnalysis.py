'''
Principal Component Analysis
'''
import numpy as np

#Function to compute the mean
def computeMean(x):
    return x.mean(0)

#Function to compute a centered data matrix
def center(x, mean):
    return x-mean

#Function to compute unnormalized covariance matrix of the centered data
def covar(data):
    dataTransposed = data.T
    dataTransposedTwo = dataTransposed
    covar = np.zeros((np.size(data,1), np.size(data,1)))
    for i, r in enumerate(dataTransposed):
        for j, c in enumerate(dataTransposedTwo):
            covar[i][j] = ((np.dot(c,r)) / (r.size-1))
    return covar

#Function to compute the first K highest principal components
def components(covar, dimension):
    #Get the eigenvalues/vectors
    eigenvalues, eigenvectors = np.linalg.eig(covar)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:,idx]    
    eigenvalues = eigenvalues[idx]
    #Get k eigenvectors corresponding to dimension k largest eigenvalues
    eigenvectors = eigenvectors[:dimension, :]
    return(eigenvectors)

#Function to compute the best K-Dimension representation of data matrix
def Representation(covarX, centerX, dimension):
    eigenvalues, eigenvectors = np.linalg.eig(covarX)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:,idx]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:dimension, :]
    return np.dot(centerX, eigenvectors.T)

#Create a function mypca which take inputs of a data matrix assuming column data, and return the best k-dimensional representation
def mypca(X, k):
    meanX = computeMean(X)
    centerX = center(X, meanX)
    covarX = covar(centerX)
    eigenvalues, eigenvectors = np.linalg.eig(covarX)
    #principal values are the k highest eigenvalues
    pv = eigenvalues[np.argsort(eigenvalues)[-k:]]
    #principal components are the k sets of eigenvectors corresponding to the k highest eigenvalues
    pc = components(covarX, k)
    #computing the k-dimension representation using our pc and pv
    rep = Representation(covarX, centerX, k)
    return pv, pc, rep

if __name__ == '__main__':
    x = np.array([[-2,1,4,6,5,3,6,2], [9,3,2,-1,-4,-2,-4,5], [0,7,-5,3,2,-3,4,6]])
    print(x)
    #Use function to compute the mean of our dataset
    meanX = computeMean(x)
    print(meanX)
    #Use function to compute centered data matrix of our dataset
    centerX = center(x, meanX)
    print(centerX)
    #Compute the covariance matrix of our centered dataset
    covarX = covar(centerX)
    print(covarX)
    #Compute the first principal component
    print(components(covarX, 1))
    #Compute the best 1D representation of data matrix
    print (Representation(covarX, centerX, 1))
    #Testing our function with our original data set
    print (mypca(x, 1))
    
    pass