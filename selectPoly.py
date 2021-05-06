import os
import sys
from numpy.lib.function_base import append
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
   # plt.show()


plot = False
try:
    if sys.argv.index("--plot") > 0:
        plot = True
except:
    plot = False



def mleFit(x,y):
    #return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return np.linalg.solve(x.T.dot(x),(x.T).dot(y))
def addBias(x):
    return np.column_stack((np.ones(x.shape),x))

def addPolyTerms(x,n):
    xs = addBias(x)
    for i in range(2,n+1):
        xs = np.column_stack((xs,x**(i)))
        #print(i)
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
    #print(A.shape)
    polyXs = addPolyTerms(testXs,n)
    #print(polyXs)
    # print("\n\n ",n)
    # print(A)
    # A,__ ,_ ,___= np.linalg.lstsq(polyXs,testYs,0)
    # print(A)
    # A = np.flip(np.polyfit(trainXs,trainYs,n))
    # print(A)
    # calcYs = np.empty(testXs.shape)
    # for i in range(n+1):
    #     calcYs += A[i]* polyXs[i]
    calcYs = np.dot(addPolyTerms(testXs,n),A)  
    diff = np.square(np.subtract(testYs, calcYs))
    error = np.sum(diff)
    if doPlot:
        newXs = np.linspace(trainXs[0],trainXs[-1],100)
        calcNewYs = np.dot(addPolyTerms(newXs,n),A) 
        plt.plot(newXs,calcNewYs,label = "Degree "+str(n))
        plt.legend(loc="best")
        view_data_segments(trainXs,trainYs)
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




weightsList = [calcLinearError, calcPolyError, calcTrigError]

def comparePoly(file,segnums,max=6):
    xs,ys = load_points_from_file(file)
    reconstructionError = np.empty(10)
    numSegs = len(xs) // 20
    for i in range(numSegs):
        if i == segnums :
            cutXs = xs[20*i:20*i+20]
            cutYs = ys[20*i:20*i+20]
            reconstructionError = [calcPolyError(cutXs,cutXs,cutYs,cutYs,True,n=j) for j in range(2,max)]
    return np.array(reconstructionError)
            

# totError = np.add(comparePoly("basic_4.csv",[1]),comparePoly("basic_3.csv",[0]))
# totError = np.add(totError,comparePoly("adv_1.csv",[0,2]))
# totError = np.add(totError,comparePoly("adv_3.csv",[1,5]))
# print(totError)
def ploterrors(totError):
    fig, ax = plt.subplots()
    ax.bar(range(2,len(totError)+2),totError)
    ax.set_xticks(np.arange(2,len(totError)+2,1))
    ax.set(xlabel='Polynomial Degree', ylabel='Sum error',
                title='Reconstruction error for different polynomial degrees')
    plt.show()


def showseg(file,segnums):
    xs,ys = load_points_from_file(file)
    #reconstructionError = np.empty(10)
    numSegs = len(xs) // 20
    for i in range(numSegs):
        if i == segnums :
            cutXs = xs[20*i:20*i+20]
            cutYs = ys[20*i:20*i+20]
            calcTrigError(cutXs,cutXs,cutYs,cutYs,True)
            view_data_segments(cutXs,cutYs)
            plt.show()

# vals =comparePoly("basic_4.csv",1)+comparePoly("basic_3.csv",0)+comparePoly("adv_1.csv",0)+comparePoly("adv_1.csv",2)+comparePoly("adv_3.csv",1)+comparePoly("adv_3.csv",5)
# vals = np.sqrt(vals)
# ploterrors(vals)
#showseg("basic_5.csv",0)
ploterrors(comparePoly("basic_5.csv",0))
ploterrors(comparePoly("basic_4.csv",1))
ploterrors(comparePoly("basic_3.csv",0))
ploterrors(comparePoly("adv_1.csv",0))
ploterrors(comparePoly("adv_1.csv",2) )
ploterrors(comparePoly("adv_3.csv",1))
ploterrors(comparePoly("adv_3.csv",5))
# showseg("adv_2.csv",0)
# showseg("adv_3.csv",2)
# showseg("adv_3.csv",4)