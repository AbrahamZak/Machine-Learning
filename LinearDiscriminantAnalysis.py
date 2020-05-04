'''
Calculates LDA on a xp and xn datasets then uses that to classify another dataset
'''
import numpy as np

def computeMean(x):
    return x.mean(0)

def covar(x, mean):
    data = x.T-mean
    dataTransposed = data.T
    dataTransposedTwo = dataTransposed
    covar = np.zeros((np.size(data,1), np.size(data,1)))
    for i, r in enumerate(dataTransposed):
        for j, c in enumerate(dataTransposedTwo):
            covar[i][j] = ((np.dot(c,r)) / (r.size-1))
    return covar

def betweenScatter (xp, xn, meanXp, meanXn):
    SizeMult = xp.shape[1]*xn.shape[1]
    SumSizeSquared = (xp.shape[1]+xn.shape[1])**2
    meansDifference = ((meanXp - meanXn))
    return (SizeMult/SumSizeSquared) * np.outer(meansDifference, meansDifference)

def withinScatter(xp, xn, covarXp, covarXn):
    xpLen = xp.shape[1]
    xnLen = xn.shape[1]
    totalLen = xpLen + xnLen
    return ((xpLen/totalLen) * covarXp) + ((xnLen/totalLen) * covarXn)

def LDAproject(Sw, Sb):
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[idx]
    eigenvalues = eigenvalues[idx]
    highestEigenvector = eigenvectors[0]
    return highestEigenvector

def mybLDA_train(xp, xn):
    meanXp = computeMean(xp.T)
    meanXn = computeMean(xn.T)
    covarXp = covar(xp, meanXp)
    covarXn = covar(xn, meanXn)
    Sb = betweenScatter(xp, xn, meanXp, meanXn)
    Sw = withinScatter(xp, xn, covarXp, covarXn)
    proj = LDAproject(Sw, Sb)
    return proj

def mybLDA_classify(x,v):
    result = []
    for row in x.T:
        res = np.dot(row,v)
        if (res>0):
            result.append(1)
        else:
            result.append(-1)
    return np.array(result)

if __name__ == '__main__':
    xp = np.array([[4,2,2,3,4,6,3,8], [1,4,3,6,4,2,2,3], [0,1,1,0,-1,0,1,0]])
    xn = np.array([[9,6,9,8,10], [10,8,5,7,8], [1,0,0,1,-1]])
    print(xp.T)
    print(xn.T)
    meanXp = computeMean(xp.T)
    meanXn = computeMean(xn.T)
    print(meanXp)
    print(meanXn)
    covarXp = covar(xp, meanXp)
    covarXn = covar(xn, meanXn)
    print(covarXp)
    print(covarXn)
    Sb = betweenScatter(xp, xn, meanXp, meanXn)
    print (Sb)
    Sw = withinScatter(xp, xn, covarXp, covarXn)
    print (Sw)
    proj = LDAproject(Sw, Sb)
    print(proj)
    train = mybLDA_train(xp,xn)
    x = np.array([[1.3,2.4,6.7,2.2,3.4,3.2], [8.1,7.6,2.1,1.1,0.5,7.4], [-1,2,3,-2,0,2]])
    classify = mybLDA_classify(x,train)
    print(classify)
    
    pass