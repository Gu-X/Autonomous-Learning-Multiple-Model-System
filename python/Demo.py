import numpy
import scipy.io
import pandas
import math
import os
from ALMMo import ALMMo1_regression_testing
# Import training function of ALMMo-1 system for classification 
from ALMMo import ALMMo1_regression_learning
# Import function of the ALMMo-1 system for classification during validation stage
##
# Read the data
dirpath = os.getcwd()
dr =dirpath+'\\data.mat'
mat_contents = scipy.io.loadmat(dr)
datain=mat_contents['data0']
dataout=mat_contents['Y0']
## Training stage
datain=numpy.matrix(datain) # Input
dataout=numpy.matrix(dataout)# The desired output
L,W=datain.shape
Estimation,SystemParam=ALMMo1_regression_learning(datain,dataout)# Train the ALMMo-1 system
# Estimation: output of the ALMMo system (estimated outputs); SystemParam: the trained ALMMo system
RMSE=numpy.sqrt(numpy.sum(numpy.power(numpy.matrix(Estimation[range(1,L)])-dataout[range(1,L),0].transpose(),2))/(L-1)) # Calculate RMSE
print (RMSE)
## Validation stage
datain=mat_contents['data1']
dataout=mat_contents['Y1']
L,W=datain.shape
Estimation=ALMMo1_regression_testing(datain,SystemParam) # Validate the trained ALMMo-1 system
# Estimation: output of the ALMMo system (estimated outputs)
RMSE=numpy.sqrt(numpy.sum(numpy.power(numpy.matrix(Estimation[range(1,L)])-dataout[range(1,L),0].transpose(),2))/(L-1)) # Calculate RMSE
print (RMSE)
