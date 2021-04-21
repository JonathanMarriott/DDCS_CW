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
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

def addBias(x):
    return np.column_stack((np.ones(x.shape),x))

def addPolyTerms(x,n):
    xs = addBias(x)
    for i in range(2,n):
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
    xtrainVals = np.delete(xs,testIndex)
    xtestVals =  xs[testIndex]
    ytrainVals = np.delete(ys,testIndex)
    ytestVals = ys[testIndex]
    return xtrainVals, xtestVals, ytrainVals, ytestVals

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
        plt.plot(testXs,calcYs)
    return error


def calcTrigError(trainXs, testXs, trainYs, testYs, doPlot=False):
    A = fitTrig(trainXs,trainYs)
    calcYs = addTrigTerms(testXs) @ A  
    diff = np.square(np.subtract(testYs, calcYs))
    error = np.sum(diff)
    if doPlot:
        plt.plot(testXs,calcYs)
    return error

def meanError(func,xs,ys):
    return np.average([func(*testTrain(xs,ys,i)) for i in range(20)])


numSegs = len(xs) // 20
#reconstructionError = []
weightsList = [calcLinearError, calcPolyError, calcTrigError]
for i in range(numSegs):
        cutXs = xs[20*i:20*i+20]
        cutYs = ys[20*i:20*i+20]
        
        # for j in range(10):
        #     reconstructionError[j] = calcPolyError(cutXs,cutXs,cutYs,cutYs,plot,n=j)
        reconstructionError = [calcPolyError(cutXs,cutXs,cutYs,cutYs,plot,n=j) for j in range(2,12)]
        fig, ax = plt.subplots()
        ax.bar(range(2,12),reconstructionError)
        ax.set_xticks(np.arange(2,len(range(12)),1))
        ax.set(xlabel='Polynomial Degree', ylabel='Sum square error',
       title='Reconstruction error for different polynomial degrees')
        plt.show()
        
        calcPolyError(cutXs,cutXs,cutYs,cutYs,True,n=3)
        view_data_segments(cutXs,cutYs)
        plt.show()

print(reconstructionError)    
if plot:
    view_data_segments(xs,ys)    

