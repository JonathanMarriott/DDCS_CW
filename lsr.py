import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

xs,ys = load_points_from_file(sys.argv[1])
plot = False
try:
    if sys.argv.index("--plot") > 0:
        plot = True
except:
    plot = False

xs = np.array(xs)
ys = np.array(ys)

def mleFit(x,y):
    return np.linalg.solve(x.T.dot(x),(x.T).dot(y))

def addBias(x):
    return np.column_stack((np.ones(x.shape),x))

def addPolyTerms(x,n):
    xs = addBias(x)
    for i in range(2,n+1):
        xs = np.column_stack((xs,x**(i)))
    return xs

def addTrigTerms(x):
    return np.column_stack((np.ones(x.shape),np.sin(x),np.cos(x)))

def fitTrig(xs,ys):
    return mleFit(addTrigTerms(xs),ys)

def fitLinear(xs,ys):
    return mleFit(addBias(xs),ys)

def fitPoly(xs,ys,n):
    return mleFit(addPolyTerms(xs,n),ys)

def testTrain(xs,ys,testIndex):
    '''Creates a test and training set with the given index specifying the testing set'''
    xtrainVals = np.delete(xs,testIndex)
    xtestVals =  xs[testIndex]
    ytrainVals = np.delete(ys,testIndex)
    ytestVals = ys[testIndex]
    return xtrainVals, xtestVals, ytrainVals, ytestVals
#
# Random sample selector  
# def testTrain(xs,ys):
#     trainIndex = random.sample(range(20),15)
#     xtrainVals = xs[trainIndex]
#     xtestVals = np.delete(xs,trainIndex)
#     ytrainVals = ys[trainIndex]
#     ytestVals = np.delete(ys,trainIndex)
#     return xtrainVals, xtestVals, ytrainVals, ytestVals

def calcLinearError(trainXs, testXs, trainYs, testYs,doPlot=False):
    A = fitLinear(trainXs,trainYs)
    calcYs = addBias(testXs) @ A  
    diff = np.square(np.subtract(testYs, calcYs))
    error = np.sum(diff)
    if doPlot:
        plt.plot(testXs,calcYs)
    return error

def calcPolyError(trainXs, testXs, trainYs, testYs,doPlot=False,n=3):
    A = fitPoly(trainXs,trainYs,n)
    calcYs = addPolyTerms(testXs,n) @ A  
    diff = np.square(np.subtract(testYs, calcYs))
    error = np.sum(diff)
    if doPlot:
        newXs = np.linspace(trainXs[0],trainXs[-1],100)
        calcNewYs = np.dot(addPolyTerms(newXs,3),A) 
        plt.plot(newXs,calcNewYs)
    return error


def calcTrigError(trainXs, testXs, trainYs, testYs, doPlot=False):
    A = fitTrig(trainXs,trainYs)
    calcYs = addTrigTerms(testXs) @ A  
    diff = np.square(np.subtract(testYs, calcYs))
    error = np.sum(diff)
    if doPlot:
        newXs = np.linspace(trainXs[0],trainXs[-1],100)
        calcNewYs = np.dot(addTrigTerms(newXs),A) 
        plt.plot(newXs,calcNewYs)
    return error


def meanError(func,xs,ys):
    '''Carries out cross validation k-fold over the inputed data and averages the total error'''
    return np.average([func(*testTrain(xs,ys,i)) for i in range(20)])


numSegs = len(xs) // 20
reconstructionError = 0
weightsDict = [calcLinearError, calcPolyError, calcTrigError]
for i in range(numSegs):
        cutXs = xs[20*i:20*i+20]
        cutYs = ys[20*i:20*i+20]
        errors = np.array([meanError(calcLinearError,cutXs,cutYs),meanError(calcPolyError,cutXs,cutYs),1.025*meanError(calcTrigError,cutXs,cutYs)])
        best = np.argmin(errors)
       # print(f'Seg Number {i} has best {["linear","poly","trig"][best]}')
        
        reconstructionError += weightsDict[best](cutXs,cutXs,cutYs,cutYs,plot)

print(reconstructionError)    
if plot:
    view_data_segments(xs,ys)    

