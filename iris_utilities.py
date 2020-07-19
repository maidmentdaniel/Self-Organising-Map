"""
Daniel Maidment

Tue Aug 13 13:38:50 2019
"""
########################################################################
import spyder_utilities as su
import numpy as np
import matplotlib.pyplot as plt

########################################################################

def readData(fname, debug=False):
    #read data file into data array
    data = []
    file = open(fname, 'r')
    for line in file:
        data_line = line.strip().split(',')
        if(debug):print(data_line)
        data.append([float(j) for j in data_line])
    file.close()
    data = np.array(data, dtype = int)
    return data

def writeData(fname, data):
    file = open(fname, 'w')
    M = np.shape(data)[0]
    N = np.shape(data)[1]
    for i in range(M):
        for j in range(N):
            file.write(str(data[i, j]))
            if(j<N-1):
                file.write(',')
        if(i<M-1):
            file.write('\n')
    file.close()
    
def normalise_data(data, debug = False):
    """
    Should normalise each feature of 'data' -not including the lables-
    between 0 and 1.
    
    
    Args:
        data [numpy array]: Shape = (number of data points, number of 
        features + integer label).
    """
    shape = np.shape(data)
    norm_data = np.zeros(shape)
    
#    mu_arr = np.mean(data[:, 0:shape[1]-1], axis = 0)
#    std_arr = np.std(data[:, 0:shape[1]-1], axis = 0)
    max_ar = np.max(data[:, 0:shape[1]-1], axis = 0)
    min_ar = np.min(data[:, 0:shape[1]-1], axis = 0)
    
    for j in range(shape[1]-1):
        norm_data[:, j] = (data[:, j]-min_ar[j])/(max_ar[j]-min_ar[j])
    
    norm_data[:, -1] = data[:, -1]
    
    if(debug):
        for i in range(shape[0]):
            if(i%(shape[0]/5)==0):
                print(data[i, :])
                print(norm_data[i, :],'\n')
    
    return norm_data