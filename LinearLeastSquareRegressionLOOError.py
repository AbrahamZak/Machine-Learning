'''
Calculates least squares regression to fit polynomials of
degree d and graphs LOO error for each degree
Outputs the one with the minimum LOO error as the model.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

if __name__ == '__main__':
    #get the x and y from data file
    x = np.genfromtxt(fname="hw2.dat", usecols=0)
    y = np.genfromtxt(fname="hw2.dat", usecols=1)
    #keep track of coefficients
    coeff = []
    #get coefficients for each degree 1-27
    for degree in range(1,28):
        z = np.polyfit(x, y, degree)
        coeff.append(z)
    #keep track of all LOO errors    
    errors = []
    #for each degree get the LOO error
    for degree in range(0,27):
        #set our sum of errors to 0
        sum_error = 0
        #for each value in x
        for val in range(0, len(x)):
            temp_x = np.delete(x,val)
            temp_y = np.delete(y,val)
            #get the mean squared error for everything in the data set except for the current data point and add that sum into our sum_error
            sum_error += np.sum((np.polyval(coeff[degree], temp_x) - temp_y)**2)
        #when exiting the loop take the sum of all errors and divide it by the length of x   
        sum_error = sum_error / len(x) 
        errors.append(sum_error)
        
    #Print the LOO error values
    for degree in range(0,27): 
        print("Degree ", degree+1 , "LOO Error: " , errors[degree])
        
    #plot all our our LOO errors for each degree
    plt.title("LOO Errors")
    plt.xlabel("Degree")
    plt.ylabel("Error")
    for degree in range(0,27): 
        plt.plot(degree+1, errors[degree], '.')
    plt.show()  
    
    lowest_error_degree_value = sys.float_info.max
    lowest_error_degree = 0
    
    for degree in range(0,27): 
        if (errors[degree] < lowest_error_degree_value):
            lowest_error_degree_value = errors[degree]
            lowest_error_degree = degree
    print("Lowest error degree: ", lowest_error_degree+1) 
    print("Lowest error: ", errors[lowest_error_degree])    
   
    plt.title("Linear least square regression (lowest LOO error) degree: " + str(lowest_error_degree+1))
    plt.xlabel("x")
    plt.ylabel("y")
    #plot original data
    plt.plot(x,y, '.', label='data')
    #plot the fit 
    plt.plot(x,np.polyval(coeff[lowest_error_degree],x), '-', label='regression')
    plt.legend();
    plt.show()                    
    pass