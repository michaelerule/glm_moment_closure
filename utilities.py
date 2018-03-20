#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
from scipy.signal.signaltools import fftconvolve,hilbert
from scipy.signal import butter, filtfilt, lfilter

# Sample from OU
def sample_ou_process(x0,sigma,tau,dt,N,ntrial=1):
    '''
    Parameters
    ---------
    x0: initial conditions
    sigma: standard deviation of driving Wiener process
    tau: exponential damping time constant
    dt: time step
    N: number of samples to draw
        
    Returns
    -------
    simulated : np.array
        time-series of sampled values
    '''
    simulated = np.zeros((N,ntrial),'float')
    x = x0*np.ones((ntrial,),'float')
    for i in range(N):
        x += -(1./tau)*x*dt + sigma * np.random.randn(ntrial) * np.sqrt(dt)
        simulated[i] = x
    return simulated

def linfilter(A,C,x,initial=None):
    '''
    Linear response filter on data $x$ for system
    
    $$
    \partial_t z = A z + C x(t)
    $$
    
    Parameters
    ----------
    A : matrix
        K x K matrix defining linear syste,
    C : matrix
        K x N matrix defining projection from signal $x$ to linear system
    x : vector or matrix
        T x N sequence of states to filter
    initial : vector
        Optional length N vector of initial filter conditions. Set to 0
        by default

    Returns
    -------
    filtered : array
        filtered data
    '''
    # initial state for filters (no response)
    L = len(x)
    K = A.shape[0]
    z = np.zeros((K,1)) if initial is None else initial
    filtered = []
    for t in range(L):
        dz = A.dot(z) + C.dot([[x[t]]])
        z += dz
        filtered.append(z.copy())
    return np.squeeze(np.array(filtered))
    
def box_filter(data,smoothat):
    '''
    Smooths data by convolving with a size smoothat box
    provide smoothat in units of frames i.e. samples (not ms or seconds)
    
    Parameters
    ----------
    x : np.array
        One-dimensional numpy array of the signal to be filtred
    window : positive int
        Filtering window length in samples
    mode : string, default 'same'
        If 'same', the returned signal will have the same time-base and
        length as the original signal. if 'valid', edges which do not
        have the full window length will be trimmed
    
    Returns
    -------
    np.array :
        One-dimensional filtered signal
    '''
    N = len(data)
    assert len(data.shape)==1
    padded = np.zeros(2*N,dtype=data.dtype)
    padded[N//2:N//2+N]=data
    padded[:N//2]=data[N//2:0:-1]
    padded[N//2+N:]=data[-1:N//2-1:-1]
    smoothed = fftconvolve(padded,np.ones(smoothat)/float(smoothat),'same')
    return smoothed[N//2:N//2+N]
    
def pulse_sequence(amplitudes,durationms,offset):
    pulses = []
    for a in amplitudes:
        for d in durationms:
            pulse = np.zeros(2*d)+offset
            pulse[d//2:d+d//2] = a
            pulses.append(pulse)
    np.random.shuffle(pulses)
    return np.array(np.concatenate(pulses))
