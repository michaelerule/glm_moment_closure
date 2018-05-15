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
from scipy.linalg import lstsq,pinv
    
import datetime
import time as systime

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
    data[~np.isfinite(data)]=0
    assert len(data.shape)==1
    padded = np.zeros(2*N,dtype=data.dtype)
    padded[N//2:N//2+N]=data
    padded[:N//2]=data[N//2:0:-1]
    padded[N//2+N:]=data[-1:N//2-1:-1]
    smoothed = fftconvolve(padded,np.ones(smoothat)/float(smoothat),'same')
    return smoothed[N//2:N//2+N]
    
def pulse_sequence(amplitudes,durationms,offset):
    '''
    Generate stimulation pulse sequences
    '''
    pulses = []
    for a in amplitudes:
        for d in durationms:
            pulse = np.zeros(2*d)+offset
            pulse[d//2:d+d//2] = a
            pulses.append(pulse)
    np.random.shuffle(pulses)
    return np.array(np.concatenate(pulses))
    
    
def linv(M,x):
    return lstsq(M,x)[0]

def cinv(x):
    '''
    Inver PSD matrix using Cholesky factorization
    '''
    ch = chol(0.5*(x+x.T)); # x = chol(x)'*chol(x)
    ch = linv(ch,np.eye(ch.shape[0]))
    return ch.dot(ch.T);

def trychol(M,reg_cov=1e-6):
    try:
        C = chol(M)
    except LinAlgError:
        M = repair_covariance(M,reg_cov)
        C = chol(M)
    return C

def current_milli_time():
    '''
    Returns the time in milliseconds
    '''
    return int(round(systime.time() * 1000))

now = current_milli_time

def today():
    '''
    Returns
    -------
    `string` : the date in YYMMDD format
    '''
    return datetime.date.today().strftime('%Y%m%d')

__GLOBAL_TIC_TIME__ = None
def tic(st=''):
    ''' 
    Similar to Matlab tic 
    stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    try:
        __GLOBAL_TIC_TIME__
        if not __GLOBAL_TIC_TIME__ is None:
            print('t=%dms'%((t-__GLOBAL_TIC_TIME__)),st)
        else: print("timing...")
    except: print("timing...")
    __GLOBAL_TIC_TIME__ = current_milli_time()
    return t

def toc(st=''):
    ''' 
    Similar to Matlab toc 
    stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    try:
        __GLOBAL_TIC_TIME__
        if not __GLOBAL_TIC_TIME__ is None:
            print('dt=%dms'%((t-__GLOBAL_TIC_TIME__)),st)
        else:
            print("havn't called tic yet?")
    except: print("havn't called tic yet?")
    return t

# For repairing numerical issues with covariance
from statsmodels.stats.correlation_tools import cov_nearest
from numpy.linalg.linalg import cholesky as chol
from numpy.linalg.linalg import LinAlgError

def repair_covariance(M2,reg_cov):
    '''
    Suppress numeric errors by keeping
    covariance matrix positive semidefiniteb
    '''
    K = M2.shape[0]
    strength = reg_cov+max(0,-np.min(np.diag(M2)))
    M2 = 0.5*(M2+M2.T) + strength*np.eye(K) 
    for retry in range(20):
        try:
            ch = chol(M2)
            break
        except LinAlgError:
            M2 += strength*np.eye(K) 
            strength *= 2
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    M2 = cov_nearest(M2,method="clipped")
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                # covariance repair failed!
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        M2 = cov_nearest(M2,method="nearest")
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    # covariance repair failed!
                    pass
    return M2
    
def bandpass_filter(data,fa=None,fb=None,
    Fs=1000.,order=4,zerophase=True,bandstop=False):
    '''
    IF fa is None, assumes lowpass with cutoff fb
    IF fb is None, assume highpass with cutoff fa
    Array can be any dimension, filtering performed over last dimension

    Parameters
    ----------
        data (ndarray): data, filtering performed over last dimension
        fa (number): low-frequency cutoff. If none, highpass at fb
        fb (number): high-frequency cutoff. If none, lowpass at fa
        order (1..6): butterworth filter order. Default 4
        zerophase (boolean): Use forward-backward filtering? (true)
        bandstop (boolean): Do band-stop rather than band-pass
    '''
    N = data.shape[-1]
    padded = np.zeros(data.shape[:-1]+(2*N,),dtype=data.dtype)
    padded[...,N//2  :N//2+N] = data
    padded[...,     :N//2  ] = data[...,N//2:0    :-1]
    padded[...,N//2+N:     ] = data[...,-1 :N//2-1:-1]
    if not fa is None and not fb is None:
        if bandstop:
            b,a = butter(order,np.array([fa,fb])/(0.5*Fs),btype='bandstop')
        else:
            b,a = butter(order,np.array([fa,fb])/(0.5*Fs),btype='bandpass')
    elif not fa==None:
        # high pass
        b,a  = butter(order,fa/(0.5*Fs),btype='high')
        assert not bandstop
    elif not fb==None:
        # low pass
        b,a  = butter(order,fb/(0.5*Fs),btype='low')
        assert not bandstop
    else: raise Exception('Both fa and fb appear to be None')
    return (filtfilt if zerophase else lfilter)(b,a,padded)[...,N//2:N//2+N]
    assert 0
