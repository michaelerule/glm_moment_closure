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
import scipy
from functions import sexp,slog

def ensemble_sample(stim,B,beta,M=100):
    '''
    Sample from an autoregressive point-process model. 
    
    Parameters
    ----------
    stim: Effective input. Incorporates filtered stimulus and constant offset
    B: History basis
    beta: Projection from the history basis onto log-rate

    Other Parameters
    ----------------
    M: number of samples to draw
    
    Returns
    -------
    y : ntime x nsamples matrix of point-process samples
    '''
    N = B.shape[1]
    T = len(stim)
    h = np.zeros((N,M))
    y = np.zeros((T,M))
    l = np.zeros((T,M))
    b = beta.T.dot(B)
    for i,s in enumerate(stim):
        l[i,:]  = s + b.dot(h)
        y[i,:]  = np.random.poisson(np.exp(l[i]))>0
        h[1:,:] = h[:-1,:]
        h[0,:]  = y[i,:]
    return y,l

def ensemble_sample_moments(stim,B,beta,M=1000):
    '''
    Estimate moments of single-time marginals of a point process using
    sampling.
    
    Parameters
    ----------
    stim: 
        Effective input. Incorporates filtered stimulus and constant offset
    B: 
        History basis
    beta: 
        Projection from the history basis onto log-rate

    Other Parameters
    ----------------
    M: number of samples to draw
    
    Returns
    -------
    allLR : single-time marginal mean of log-rate
    allLV : single-time marginal variance of log-rate
    allLR : single-time marginal mean of rate
    allLV : single-time marginal variance of rate
    '''
    N = B.shape[1]
    T = len(stim)
    h = np.zeros((N,M))
    y = np.zeros((T,M))
    l = np.zeros((T,M))
    b = beta.T.dot(B)
    for i,s in enumerate(stim):
        l[i,:]  = s + b.dot(h)
        y[i,:]  = np.random.poisson(np.exp(l[i]))>0
        h[1:,:] = h[:-1,:]
        h[0,:]  = y[i,:]
    logm = np.mean(l,axis=1)
    logv = np.var(l,axis=1)
    expm = np.mean(np.exp(l),axis=1)
    expv = np.var(np.exp(l),axis=1)
    return logm,logv,expm,expv


def langevin_sample(stim,A,beta,C,
                    dt         = 1.0,
                    M          = 10000,
                    maxrate    = 4,
                    oversample = 4):
    '''
    Sample from the Langevin approximation of the autoregressive point-process model. 
    History filters are replaced by linear filters, defined by the operators A, C, and
    beta. Poisson noise is replaced with Gaussian noise with variance equal to the mean
    rate. 
    
    Parameters
    ----------
    stim: Effective input. Incorporates filtered stimulus and constant offset
    A: Forward linear operator for the history filter
    beta: Projection from the history filter onto log-rate
    C: Projection from the point-process onto the linear history filter
    
    Other Parameters
    ----------------
    dt: time step
    M: number of samples to draw
    maxrate: maximum rate
    oversample: sampling resolution
    
    Returns
    -------
    allLR : single-time marginal mean of log-rate
    allLV : single-time marginal variance of log-rate
    allLR : single-time marginal mean of rate
    allLV : single-time marginal variance of rate
    '''
    maxlograte = np.log(maxrate)
    dtfine     = dt/oversample
    T   = len(stim)
    K   = beta.size
    Adt = A*dtfine
    z   = np.zeros((K,M))
    l   = np.zeros((T,M))
    for i,s in enumerate(stim):
        for j in range(oversample):
            lr = np.minimum(s + beta.T.dot(z),maxlograte)
            r = np.exp(lr)*dtfine
            Y = np.random.randn(M)*np.sqrt(r) + r
            z = z + Adt.dot(z) + C*Y
        l[i] = lr
    return l

def langevin_sample_moments(stim,A,beta,C,
                    dt         = 1.0,
                    M          = 10000,
                    maxrate    = 4,
                    oversample = 4):
    '''
    Sample from the Langevin approximation of the autoregressive point-process model. 
    History filters are replaced by linear filters, defined by the operators A, C, and
    beta. Poisson noise is replaced with Gaussian noise with variance equal to the mean
    rate. 
    
    Parameters
    ----------
    stim: Effective input. Incorporates filtered stimulus and constant offset
    A: Forward linear operator for the history filter
    beta: Projection from the history filter onto log-rate
    C: Projection from the point-process onto the linear history filter
    
    Other Parameters
    ----------------
    dt: time step
    M: number of samples to draw
    maxrate: maximum rate
    oversample: sampling resolution
    
    Returns
    -------
    allLR : single-time marginal mean of log-rate
    allLV : single-time marginal variance of log-rate
    allLR : single-time marginal mean of rate
    allLV : single-time marginal variance of rate
    '''
    maxlograte = np.log(maxrate)
    dtfine     = dt/oversample
    T   = len(stim)
    K   = beta.size
    Adt = A*dtfine
    z   = np.zeros((K,M))
    l   = np.zeros((T,M))
    for i,s in enumerate(stim):
        for j in range(oversample):
            lr = np.minimum(s + beta.T.dot(z),maxlograte)
            r = np.exp(lr)*dtfine
            Y = np.random.randn(M)*np.sqrt(r) + r
            z = z + Adt.dot(z) + C*Y
        l[i] = lr
    logm = np.mean(l,axis=1)
    logv = np.var(l,axis=1)
    expm = np.mean(np.exp(l),axis=1)
    expv = np.var(np.exp(l),axis=1)
    return logm,logv,expm,expv
    

def langevin_sample_expected(stim,A,beta,C,
                    dt         = 1.0,
                    M          = 100,
                    maxrate    = 4,
                    oversample = 4):
    maxlograte = np.log(maxrate)
    dtfine     = dt/oversample
    T   = len(stim)
    K   = beta.size
    Adt = A*dtfine
    z   = np.zeros((K,M))
    allz= np.zeros((T,K,M))
    l   = np.zeros((T,M))
    for i,s in enumerate(stim):
        for j in range(oversample):
            lr = np.minimum(s + beta.T.dot(z),maxlograte)
            r = np.exp(lr)*dtfine
            Y = np.random.randn(M)*np.sqrt(r) + r
            z = z + Adt.dot(z) + C*Y
        l[i] = lr
        allz[i] = z
    return l,allz
    
    
from scipy.special import polygamma
def inversepolygamma(n,x,iterations=1500,tol=1e-16):
    '''
    The Gaussian moment closure computes <λ> and <λh> under the assumption
    that λ is log-Gaussian distributed, with the moments given by the
    moment-closure equations. 
    
    And extension of this is to allow the moments to describe the moments
    of an arbitrary exponential-family distribution. 
    
    For example, the distribution of λ is typically less right-skewed than
    predicted by the log-Gaussian model. One such distribution, defined 
    on the positive real numbers, that exhibits less skewness is the
    Gamma distribution.
    
    It is possible to match the moments of the log-Gaussian to the moments
    of the Gamma distribution (transformed to a distribution on ln λ). 
    This is sometimes called the "exp-gamma" distribution.
    
    This distributional assumption can then be used to compute expectations.
    This idea is quite general, and the Gamma assumption didn't turn out to
    be all that more accurate in practice.
    '''
    x = np.maximum(x,1e-19)
    x = np.array(x)
    if np.prod(x.shape)>1:
        y = [inversepolygamma(n,xi,iterations,tol) for xi in x.ravel()]
        y = np.array(y).reshape(x.shape)
        return y
    y = 1/x
    i = 0
    while i<iterations:
        newy = y-(polygamma(n,y)-x)/polygamma(n+1,y)
        if abs(newy-y)<tol:
            break
        y = newy
        i+=1
    return y
    
def integrate_moments(stim,A,beta,C,
    dt         = 1.0,
    oversample = 25,
    maxrate    = 500,
    maxvcorr   = 2000,
    method     = "moment_closure",
    int_method = "euler"):
    '''
    
    Parameters
    ----------
    stim : zero-lage effective input (filtered stimulus plus mean offset)
    A : forward operator for delay-line evolution
    C : projection of current state onto delay-like
    beta : basis history weights
    
    Other Parameters
    ----------------
    dt : time step
    oversample : int
        Integration steps per time step. Should be larger if using 
        Gaussian moment closure, which is stiff. Can be small if using
        second-order approximations, which are less stiff.
    maxrate : 
        maximum rate tolerated
    maxvcorr: 
        Maximum variance correction ('convexity correction' in some literature)
        tolerated during the moment closure.
    method : 
        Moment-closure method. Can be "LNA" for mean-field with linear
        noise approximation, "moment_closure" for Gaussian moment-closure
        on the history process, or "second_order", which discards higher
        moments of the rate which emerge when exponentiating.
    int_method:
        Integration method. Can be either "euler" for forward-Euler, or 
        "exponential", which integrates the locally-linearized system 
        forward using matrix exponentiation (slower).
    
    Returns
    -------
    allLR : single-time marginal mean of log-rate
    allLV : single-time marginal variance of log-rate
    allM1 : low-dimensional approximation of history process, mean
    allM2 : low-dimensional approximation of history process, covariance
    '''
    maxlogr   = np.log(maxrate)
    maxratemc = maxvcorr*maxrate
    dtfine    = dt/oversample
    T         = len(stim)
    K         = beta.size
    
    # Precompute constants
    Cb  = C.dot(beta.T)
    CC  = C.dot(C.T)
    Adt = A*dtfine
    F1  = scipy.linalg.expm(Adt)
    
    # Initial condition for moments
    M1 = np.zeros((K,1))
    M2 = np.eye(K)*1e-6
    
    if int_method=="exponential":
        # Integrate locally linearized system 
        # via exponentiation
        mean_update = lambda M1: F1.dot(M1)
        def cov_update(M2,J):
            F2 = scipy.linalg.expm(J)
            return F2.dot(M2).dot(F2.T)
    
    elif int_method=="euler":
        # Integrate locally linearized system 
        # via forward Euler method
        mean_update = lambda M1: M1 + Adt.dot(M1)
        def cov_update(M2,J):
            JM2 = J.dot(M2)
            return M2 + JM2 + JM2.T
    
    allM1 = np.zeros((T,K))
    allM2 = np.zeros((T,K,K))
    allLR = np.zeros((T))
    allLV = np.zeros((T))
    
    if method=="LNA":
        def update(logx,logv,Rm,M1,M2):
            return Rm,Cb*Rm+Adt
    elif method=="second_order":
        def update(logx,logv,R0,M1,M2):
            Rm = R0 * min(1+0.5*logv,maxvcorr)
            return Rm,Cb*R0+Adt
    elif method=="moment_closure":
        def update(logx,logv,R0,M1,M2):
            Rm = R0 * min(sexp(0.5*logv),maxvcorr)
            return Rm,Cb*Rm+Adt
    for i,s in enumerate(stim):
        for j in range(oversample):
            assert(np.all(np.isfinite(M1)) and np.all(np.isfinite(M2)))
            logv  = beta.T.dot(M2).dot(beta)
            logx  = min(beta.T.dot(M1)+s,maxlogr)
            R0    = sexp(logx)*dtfine
            Rm,J  = update(logx,logv,R0,M1,M2)
            M2    = cov_update(M2,J) + CC*Rm
            M1    = mean_update(M1) + C*Rm
        allM1[i] = M1[:,0]
        allM2[i] = M2
        allLR[i] = logx
        allLV[i] = beta.T.dot(M2).dot(beta)

    return allLR,allLV,allM1,allM2
