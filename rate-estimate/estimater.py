import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

mpl.rcParams['font.size'] = 20
mpl.rcParams['font.family'] = 'TimesNewRowman'
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
class Parameters(object):

    def __init__(self, transitionMatrix, rateMatrix, lagtime, p0, maxTime):
        self.p0 = p0
        self.maxTime = maxTime
        self.lagtime = lagtime
        self.rateMatrix = rateMatrix
        self.transitionMatrix = transitionMatrix
        self.yRef = self.__gain_population_from_transition_matrix(transitionMatrix, p0, lagtime, maxTime)
        self.weights, self.index, self.shape = self.rate_matrix_to_weights(self.rateMatrix)

    def weights_to_rate_matrix(self, weights, index=None, shape=None):
        if index is None:
            index = self.index
        if shape is None:
            shape = self.shape
        rateMatrix = np.zeros(shape)
        for i, (col, row) in enumerate(index):
            rateMatrix[col, row] = weights[i]
        return rateMatrix

    def rate_matrix_to_weights(self, rateMatrix, precision=1e-30):
        weights, index = [], []
        Ncol, Nrow = rateMatrix.shape
        for col in range(Ncol):
            for row in range(Nrow):
                if rateMatrix[col,row] - 0 > precision:
                    weights.append(rateMatrix[col, row])
                    index.append([col, row])
        return np.array(weights), index, rateMatrix.shape

    def __gain_population_from_transition_matrix(self, transitionMatrix, p0, lagtime, maxTime):
        populations = [p0]
        iterN = int(maxTime/lagtime)
        for i in range(iterN-1):
             populations.append(np.dot(populations[-1], transitionMatrix))
        populations = np.array(populations)
        return populations

    def gain_population_from_rate_matrix(self, rateMatrix, p0=None, lagtime=None, maxTime=None):
        if p0 is None:
            p0 = self.p0
        if lagtime is None:
            lagtime = self.lagtime
        if maxTime is None:
            maxTime = self.maxTime
        times = np.arange(0, maxTime, lagtime)
        populations = [p0]
        for i in times[1:]:
            p = []
            for i in range(len(p0)):
                transout = sum(p0[i] * rateMatrix[i,:]) * lagtime
                transin = np.dot(p0, rateMatrix[:,i]) * lagtime
                p.append(p0[i]-transout+transin)
            p0 = p
            populations.append(p0)
        populations = np.array(populations)
        return populations

def square_loss(y0, y1):
    assert len(y0)==len(y1), 'not the same shape y0: %d, y1:%d'%(len(y0), len(y1))
    return np.sum((y1-y0)**2)/len(y0)

def abs_loss(y0, y1):
    assert len(y0)==len(y1), 'not the same shape y0: %d, y1:%d'%(len(y0), len(y1))
    return np.sum(np.abs(y1-y0))/len(y0)

def loss_function(weights, paras):
    rateMatrix = paras.weights_to_rate_matrix(weights)
    yPredict = paras.gain_population_from_rate_matrix(rateMatrix)
    yError = square_loss(np.concatenate(paras.yRef), np.concatenate(yPredict))
    # print(yError)
    # yError = abs_loss(np.concatenate(paras.yRef), np.concatenate(yPredict))
    # print(weights)
    # print(yError)
    return yError

def update_weight(weights, wi, paras, learningRatio=0.05, deltaRatio=0.001):
    # δwi
    deltaW = weights[wi] * deltaRatio
    # print("DEBUG",weights)

    # L(wi + δwi), L(wi - δwi)
    weights0, weights1 = weights.copy(), weights.copy()
    weights0[wi], weights1[wi] = weights[wi] + deltaW, weights[wi] - deltaW

    # L(wi + δwi) - L(wi - δwi)
    right = loss_function(weights0, paras)
    left = loss_function(weights1, paras)

    deltaL = left - right
    partial = deltaL/(2 * deltaW)
    learningRate = partial/abs(partial) * weights[wi] * learningRatio
    # print("DEBUG for wights %s"%wi, deltaW, deltaL, partial, learningRate)
    # print('DEBUG: befor update: ',weights[wi])
    weights[wi] = weights[wi] + learningRate
    # print('DEBUG: after update: ',weights[wi])
    # print()
    return weights[wi], weights.copy()

def gradient_descent(paras, learningRatio=0.05, iterN=10000, epsilon=1e-6):
    iterIndex = 0
    weightsList = [paras.weights]
    lossValues = [loss_function(paras.weights, paras)]
    print('start loss: %.15f'%lossValues[-1])
    while 1:
        weights0 = weightsList[-1]
        #print(weights)
        weights = []
        for wi in range(len(weights0)):
            weightsWi, _ = update_weight(weights0, wi, paras, learningRatio)
            weights.append(weightsWi)
        weights = np.array(weights)
        #print(weights)
        lossValue = loss_function(weights, paras)
        iterIndex += 1
        if lossValue < lossValues[-1]:
            lossValues.append(lossValue)
            weightsList.append(weights)
        else:
            weights = weights0.copy()
            print("iter %6d : loss %.15f, reject, delta %.15f"%(iterIndex, lossValue,lossValues[-1]))
            #sys.stdout.write("iter %6d : loss %.15f, reject"%(iterIndex, lossValue))
            if iterIndex >= iterN:
                break
            continue
        delta = abs(lossValue - lossValues[-2])
        print("iter %6d : delta %.15f, loss %.10f, acceept"%(iterIndex, delta, lossValue))
        #sys.stdout.write("iter %6d : delta %.15f, loss %.10f, acceept"%(iterIndex, delta, lossValue))
        #sys.stdout.flush()
        if delta <= epsilon or iterIndex>=iterN:
            break
    #sys.stdout.flush()
    rateBest = paras.weights_to_rate_matrix(weightsList[-1])
    return lossValues, weightsList, rateBest

def weights_to_rate_matrix(weights, index, shape):
    rateMatrix = np.zeros(shape)
    for i, (col, row) in enumerate(index):
        rateMatrix[col, row] = weights[i]
    return rateMatrix

def rate_matrix_to_weights(rateMatrix, precision=1e-30):
    weights, index = [], []
    Ncol, Nrow = rateMatrix.shape
    for col in range(Ncol):
        for row in range(Nrow):
            if rateMatrix[col,row] - 0 > precision:
                weights.append(rateMatrix[col, row])
                index.append([col, row])
    return weights, index, rateMatrix.shape

def init_rate_matrix():
    pass

def gain_population_from_transition_matrix(transitionMatrix, p0, lagtime, maxTime):
    populations = [p0]
    iterN = int(maxTime/lagtime)
    for i in range(iterN-1):
         populations.append(np.dot(populations[-1],transitionMatrix))
    populations = np.array(populations)
    return populations

def gain_population_from_rate_matrix(rateMatrix, p0, lagtime, maxTime):
    times = np.arange(0 , maxTime, lagtime)
    populations = [p0]
    for i in times[1:]:
        p = []
        for i in range(len(p0)):
            transout = sum(p0[i] * rateMatrix[i,:]) * lagtime
            transin = np.dot(p0, rateMatrix[:,i]) * lagtime
            p.append(p0[i]-transout+transin)
        p0 = p
        populations.append(p0)
    populations = np.array(populations)
    return populations

def gain_population_from_rate_weights(weights, paras):
    index, shape, p0, lagtime, maxTime = paras.index, paras.shape, paras.p0, paras.lagtime, paras.maxTime
    rateMatrix = weights_to_rate_matrix(weights, index, shape)
    return gain_population_from_rate_matrix(rateMatrix, p0, lagtime, maxTime)

def plot_evolution(times, population, ax=None, linefmt='-', linewidth=4):
    fig = plt.gcf()
    if not ax:
        fig, ax = plt.subplots()
    nstates = population.shape[1]
    colors = list(mcolors.TABLEAU_COLORS.values())*round(nstates/10+1)
    for i in range(nstates):
        ax.plot(times, population[:,i], linefmt, linewidth=linewidth, color=colors[i])
    ax.set_ylabel('population')
    ax.set_xlabel('time')

    return fig, ax

def plot_compare(times, population0, population1, ax=None):
    fig = plt.gcf()
    if not ax:
        fig, ax = plt.subplots()
    plot_evolution(times, population0, ax, linefmt='-')
    plot_evolution(times, population1, ax, linefmt='--')

    return fig, ax
