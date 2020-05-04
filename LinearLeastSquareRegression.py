import numpy as np
import matplotlib.pyplot as plt

def plot(x,y,z, degree):
    plt.title("Linear least square regression degree: " + str(degree))
    plt.xlabel("x")
    plt.ylabel("y")
    #plot original data
    plt.plot(x,y, '.', label='data')
    #plot the fit 
    plt.plot(x,np.polyval(z,x), '-', label='regression')
    plt.legend();
    plt.show()
    
#get the polynomial from x, y, and degree
def polynomial (x, y, degree):
    z = np.polyfit(x, y, degree)
    print("Parameters of polynomial of degree ", degree, ": ", z)
    print ("Fitting error: ", np.sum((np.polyval(z, x) - y)**2))
    plot(x,y,z, degree)
    
    
if __name__ == '__main__':
    #read text file
    x = np.genfromtxt(fname="hw2.dat", usecols=0)
    y = np.genfromtxt(fname="hw2.dat", usecols=1)
    #set degrees
    degrees = {1,3,5,7}
    for degree in degrees:
        polynomial(x,y,degree)
    pass

