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
    cutXs = xs[:15]
    cutYs = ys[:15]
    testXs = xs[15:]
    testYs = ys[15:]
    A = fitLinear(cutXs,cutYs)
    calcYs = addBias(testXs) @ A  
    #plt.vlines(testXs,testYs,calcYs,colors='black')
    diff = np.absolute(np.subtract(testYs, calcYs))
    error = np.sum(diff)
   # plt.plot(xs,addTrigTerms(xs) @ A,c='red')
    #plt.scatter(cutXs,cutYs)
    #plt.scatter(testXs,testYs,marker='x',c='orange')
    return error

def calcPolyError(xs,ys,n):
    cutXs = xs[:15]
    cutYs = ys[:15]
    testXs = xs[15:]
    testYs = ys[15:]
    A = fitPoly(cutXs,cutYs,n)
    calcYs = addPolyTerms(testXs,n) @ A  
    #plt.vlines(testXs,testYs,calcYs,colors='black')
    diff = np.absolute(np.subtract(testYs, calcYs))
    error = np.sum(diff)
   # plt.plot(xs,addTrigTerms(xs) @ A,c='red')
    #plt.scatter(cutXs,cutYs)
    #plt.scatter(testXs,testYs,marker='x',c='orange')
    return error


def calcTrigError(xs,ys):
    cutXs = xs[:15]
    cutYs = ys[:15]
    testXs = xs[15:]
    testYs = ys[15:]
    A = fitTrig(cutXs,cutYs)
    calcYs = addTrigTerms(testXs) @ A  
    #plt.vlines(testXs,testYs,calcYs,colors='black')
    diff = np.absolute(np.subtract(testYs, calcYs))
    error = np.sum(diff)
   # plt.plot(xs,addTrigTerms(xs) @ A,c='red')
    #plt.scatter(cutXs,cutYs)
    #plt.scatter(testXs,testYs,marker='x',c='orange')
    return error


def plotLinear(xs,ys):
    A = fitLinear(xs,ys)
    calcYs = addBias(xs) @ A
    plt.plot(xs,calcYs)
    diff = np.absolute(np.subtract(ys, calcYs))
    return np.sum(diff)

def plotPoly(xs,ys,n=3):
    A = fitPoly(xs,ys,n)
    calcYs = addPolyTerms(xs,n) @ A
    plt.plot(xs,calcYs)
    diff = np.absolute(np.subtract(ys, calcYs))
    return np.sum(diff)

def plotTrig(xs,ys):
    A = fitTrig(xs,ys)
    calcYs = addTrigTerms(xs) @ A
    plt.plot(xs,calcYs)
    diff = np.absolute(np.subtract(ys, calcYs))
    return np.sum(diff)
# print(calcLinearError(xs,ys))
# print(calcTrigError(xs,ys))
# for i in range(1,5):
#     print(calcPolyError(xs,ys,i))
#

# numSegs = len(xs) // 20
# for i in range(numSegs):
#         cutXs = xs[20*i:20*i+20]
#         cutYs = ys[20*i:20*i+20]
#         print(calcTrigError(cutXs,cutYs))
# plt.show()


numSegs = len(xs) // 20
reconstructionError = 0
weightsDict = {0:plotLinear, 1:plotPoly,2:plotTrig}
for i in range(numSegs):
        cutXs = xs[20*i:20*i+20]
        cutYs = ys[20*i:20*i+20]
        errors = np.array([calcLinearError(cutXs,cutYs),calcPolyError(cutXs,cutYs,3),calcTrigError(cutXs,cutYs)])
        print(errors)
        
        best = np.argmin(errors)
        print(f'Seg Num: {i} has best {best}')
        
        #print(weightsDict[best](cutXs,cutYs))
        reconstructionError += weightsDict[best](cutXs,cutYs)
        
    
print(reconstructionError)
view_data_segments(xs,ys)