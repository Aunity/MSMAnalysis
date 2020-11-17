#!/usr/bin/env python
'''
@Author ymh
@Email  maohuay@hotmail.com
@Date   2020-11-17 15:17:56
@Web    https://github.com/Aunity
'''

import sys
import warnings
import numpy as np
from estimater import *

warnings.filterwarnings('ignore')
def main():
    rateInit = np.array([[0.000, 0.500, 0.000, 0.500, 0.000],
                         [0.501, 0.000, 0.000, 0.500, 0.000],
                         [0.000, 0.000, 0.000, 0.000, 0.000],
                         [0.000, 0.000, 0.500, 0.000, 0.000],
                         [0.500, 0.521, 0.000, 0.000, 0.000]])
    # transition matrix
    T = np.array([[8.18181818e-01, 9.67117988e-03, 0.00000000e+00, 1.72147002e-01, 0.00000000e+00],
                 [1.41113654e-02, 6.92601068e-01, 0.00000000e+00, 2.93287567e-01, 0.00000000e+00],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                 [4.06283857e-03, 8.47778982e-03, 5.48483207e-02, 9.32611051e-01, 0.00000000e+00],
                 [9.21100000e-03, 9.43359375e-02, 3.90625000e-04, 5.07812500e-03, 8.90984312e-01]])

    # init population
    p0 = np.array([0,0,0,0,1])

    # lagtime and maxmiun time
    lagtime, maxTime = 1, 200

    paras = Parameters(T, rateInit, lagtime, p0, maxTime)

    epsilon = 1e-10
    learningRatio = 0.005
    iterN = 1000
    lossValues, weightsList, rateBest = gradient_descent(paras, iterN=iterN, epsilon=epsilon, learningRatio=learningRatio)

    populationRef = gain_population_from_transition_matrix(T, p0, lagtime, maxTime)
    populationInit = gain_population_from_rate_matrix(rateInit, p0, lagtime, maxTime)
    populationBest = gain_population_from_rate_matrix(rateBest, p0, lagtime, maxTime)

    np.savetxt('lossValues.txt', lossValues)
    np.savetxt('rateBest.txt', rateBest)

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(6,8))
    times = np.arange(0, maxTime, lagtime)
    _, ax = plot_compare(times, populationRef, populationInit, ax=axs[0])
    ax.set_ylim(0,1)
    ax.set_title('init rate')
    _, ax = plot_compare(times, populationRef, populationBest, ax=axs[1])
    ax.set_ylim(0,1)
    ax.set_title('estimated best rate')

    fig.tight_layout()
    fig.savefig('evolution.png', dpi=100)

if __name__ == '__main__':
    main()
