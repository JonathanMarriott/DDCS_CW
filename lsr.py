import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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



xs = np.array(xs)
ys = np.array(ys)
#print(x,y)


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

def calcLinearError(xs,ys):
    numSegs = len(xs) // 20
    error = 0
    for i in range(numSegs):
        cutXs = xs[20*i:20*i+20]
        cutYs = ys[20*i:20*i+20]
        A = fitLinear(cutXs,cutYs)
        diff = np.absolute(np.subtract(cutYs, A[0]+cutXs*A[1]))
        error = error + np.sum(diff)
        #plt.plot(cutXs,A[0]+cutXs*A[1])
    return error

def calcPolyError(xs,ys,n):
    numSegs = len(xs) // 20
    error = 0
    for i in range(numSegs):
        cutXs = xs[20*i:20*i+20]
        cutYs = ys[20*i:20*i+20]
        A = fitPoly(cutXs,cutYs,n)
        calcYs = addPolyTerms(cutXs,n) @ A  
        diff = np.absolute(np.subtract(cutYs, calcYs))
        error = error + np.sum(diff)
        #plt.plot(cutXs,calcYs)
    return error


def calcTrigError(xs,ys):
    numSegs = len(xs) // 20
    error = 0
    for i in range(numSegs):
        cutXs = xs[20*i:20*i+20]
        cutYs = ys[20*i:20*i+20]
        A = fitTrig(cutXs,cutYs)
        calcYs = addTrigTerms(cutXs) @ A  
        diff = np.absolute(np.subtract(cutYs, calcYs))
        error = error + np.sum(diff)
        plt.plot(cutXs,calcYs)
    return error

print(calcLinearError(xs,ys))
print(calcTrigError(xs,ys))
for i in range(1,5):
    print(calcPolyError(xs,ys,i))
view_data_segments(xs,ys)

