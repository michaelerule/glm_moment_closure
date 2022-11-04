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
from   functions import sexp,slog

def build_depth_shifted_priors_source(M1,M2,D):
    '''
    Use existing filtered moments M1 and M2 as initial conditions for a 
    depth D parallel filtering. 
    
    Parameters
    ----------
    M1 : NxKx1 initial conditions for means
    M2 : NxKxK initial conditions for covariance
    D : depth of parallel filtering to be computed
    
    Returns
    -------
    iniM1 : initial conditions for means for a depth-D parallel filtering
    iniM2 : initial condition for covariances for depth-D parallel filter
    '''
    K = M2.shape[-1]
    N = M1.shape[0]
    defaultM1   = np.zeros((K,1))
    defaultM2   = np.eye(K)*1e-6
    iniM1       = np.zeros((N,K,1))
    iniM2       = np.zeros((N,K,K))
    iniM1[:D-1] = defaultM1
    iniM2[:D-1] = defaultM2
    iniM1[D-1:] = M1[:-D+1]
    iniM2[D-1:] = M2[:-D+1]
    return iniM1,iniM2

def filter_moments_dstep(D,S,Y,A,beta,C,m,
    dt          = 1.0,
    oversample  = 10,
    maxrate     = 1,
    maxvcorr    = 2000,
    method      = "second_order",
    int_method  = "euler",
    measurement = "moment",
    reg_cov     = 0.01,
    reg_rate    = 0.001,
    use_surrogates    = None,
    prior = None):
    '''
    Parallelized depth-D filtering
    '''
    
    if not method=='second_order':
        raise ValueError('Only the second order moment method supported for parallel filtering')
    if not int_method=='euler':
        raise ValueError('Only Euler integration supported for parallel filtering')
    if not measurement=='moment':
        raise ValueError('Onle moment-matching measurement update supported for parallel filtering')
    
    # Precompute constants
    maxlogr   = np.log(maxrate)
    dtfine    = dt/oversample
    N         = np.size(S)
    K         = beta.size
    Cb        = C.dot(beta.T)
    CC        = C.dot(C.T)
    Adt       = A*dtfine

    allLR = np.zeros(N)
    allLV = np.zeros(N)
    allM1 = np.zeros((N,K,1))
    allM2 = np.zeros((N,K,K))
    allRC = np.zeros((N,K,K))

    for i in range(N):
        allRC[i,...]=reg_cov*np.eye(K)

    if prior is None:
        iniM1 = np.zeros((N,K,1))
        iniM2 = np.zeros((N,K,K))
        for i in range(N):
            # Initial condition for moments
            iniM1[i,...]=np.zeros((K,1))
            iniM2[i,...]=np.eye(K)*1e-6
    else:
        iniM1,iniM2 = map(np.array,prior)
        
    allM1[...] = iniM1
    allM2[...] = iniM2
    # Integration range, in standard deviations,
    # for univariate moment-based update
    intr = np.linspace(-4,4,25)

    for di in range(-D+1,1):
        # Reset values that really shouldn't be being integrated? 
        allM1[:-di,...]=iniM1[:-di,...]
        allM2[:-di,...]=iniM2[:-di,...]
        # Regularize
        if reg_cov>0:
            allM2 = 0.5*(allM2 + allM2.transpose(0,2,1)) + allRC
        offsets = np.maximum(0,np.arange(N)+di)
        S_ = S[offsets]
        Y_ = Y[offsets]
        for k in range(oversample):
            LOGV = allM2.dot(beta[:,0]).dot(beta[:,0])
            LOGX = np.minimum(maxlogr,allM1[:,:,0].dot(beta[:,0])+S_)
            R0_  = np.minimum(maxrate,sexp(LOGX))*dtfine
            RM = R0_ * np.minimum(1+0.5*LOGV,maxvcorr)
            J_   = Cb[None,:,:]*R0_[:,None,None]+Adt[None,:,:]
            allM1 += np.matmul(Adt,allM1[:,:,:])
            JM2_   = np.matmul(J_,allM2)
            allM2 += JM2_ + JM2_.transpose((0,2,1))
            allM1 +=  C[None,:,:]*RM[:,None,None]
            allM2 += CC[None,:,:]*RM[:,None,None]
        # Parallel measurement update
        M2B_ = np.matmul(allM2,beta)
        LV = allM2.dot(beta[:,0]).dot(beta[:,0])
        LV = np.maximum(1e-12,LV)
        LM = allM1[:,:,0].dot(beta[:,0])
        LT = 1/LV
        TQ = LT + reg_rate
        VQ = 1/TQ
        MQ = (LM*LT+m*reg_rate)*VQ
        intr = np.linspace(-4,4,25)
        X_ = intr[None,:]*np.sqrt(VQ)[:,None]+MQ[:,None]
        R0_ = X_ + S_[:,None]+slog(dt)
        L = Y_[:,None]*R0_-sexp(R0_)
        L = L - np.max(L,axis=1)[:,None]
        L += -.5*((intr**2)[None,:]+slog(VQ)[:,None])
        PR = sexp(L)
        PR = np.maximum(1e-7,PR)
        NR = 1/np.sum(PR,axis=1)
        MP = np.sum(X_*PR,axis=1)*NR
        VP = np.sum((X_-MP[:,None])**2*PR,axis=1)*NR
        VP = np.maximum(1e-12,VP)
        TP = 1/VP
        TR = TP-LT
        TR = np.maximum(1e-12,TR)
        VR = 1/TR
        MR = (MP*TP-LM*LT)*VR
        KG = M2B_/(VR+LV)[:,None,None]
        allM2 -= np.matmul(KG,M2B_.transpose(0,2,1))
        allM1 += KG*(MR-LM)[:,None,None]
        LOGR = MP+S_
        LOGPYX = Y_*LOGR-sexp(LOGR)
        LL = LOGPYX - 0.5*(slog(LV/VP) + (MP-LM)**2/LV)
    for i in range(N):
        M1 = allM1[i,:,0]
        M2 = allM2[i]
        allLR[i] = min(beta.T.dot(M1)+S[i],maxlogr)
        allLV[i] = beta.T.dot(M2).dot(beta)

    LL = LL[np.isfinite(LL)]
    return allLR,allLV,allM1,allM2,-np.nanmean(LL)

def filter_moments_dstep_surrogate(D,S,Y,allMR,allVR,A,beta,C,m,
    dt          = 1.0,
    oversample  = 10,
    maxrate     = 1,
    maxvcorr    = 2000,
    method      = "second_order",
    int_method  = "euler",
    measurement = "moment",
    reg_cov     = 0.01,
    reg_rate    = 0.001,
    use_surrogates    = None,
    prior = None):
    '''
    Parallelized depth-D filtering
    '''
    
    if not method=='second_order':
        raise ValueError('Only the second order moment method supported for parallel filtering')
    if not int_method=='euler':
        raise ValueError('Only Euler integration supported for parallel filtering')
    if not measurement=='moment':
        raise ValueError('Onle moment-matching measurement update supported for parallel filtering')
    
    # Precompute constants
    maxlogr   = np.log(maxrate)
    dtfine    = dt/oversample
    N         = np.size(S)
    K         = beta.size
    Cb        = C.dot(beta.T)
    CC        = C.dot(C.T)
    Adt       = A*dtfine

    allLR = np.zeros(N)
    allLV = np.zeros(N)
    allM1 = np.zeros((N,K,1))
    allM2 = np.zeros((N,K,K))
    allRC = np.zeros((N,K,K))

    for i in range(N):
        allRC[i,...]=reg_cov*np.eye(K)

    if prior is None:
        iniM1 = np.zeros((N,K,1))
        iniM2 = np.zeros((N,K,K))
        for i in range(N):
            # Initial condition for moments
            iniM1[i,...]=np.zeros((K,1))
            iniM2[i,...]=np.eye(K)*1e-6
    else:
        iniM1,iniM2 = map(np.array,prior)
        
    allM1[...] = iniM1
    allM2[...] = iniM2
    # Integration range, in standard deviations,
    # for univariate moment-based update
    intr = np.linspace(-4,4,25)

    for di in range(-D+1,1):
        # Reset values that really shouldn't be being integrated? 
        allM1[:-di,...]=iniM1[:-di,...]
        allM2[:-di,...]=iniM2[:-di,...]
        # Regularize
        if reg_cov>0:
            allM2 = 0.5*(allM2 + allM2.transpose(0,2,1)) + allRC
        offsets = np.maximum(0,np.arange(N)+di)
        S_ = S[offsets]
        Y_ = Y[offsets]
        MR = allMR[offsets]
        VR = allVR[offsets]
        for k in range(oversample):
            LOGV = allM2.dot(beta[:,0]).dot(beta[:,0])
            LOGX = np.minimum(maxlogr,allM1[:,:,0].dot(beta[:,0])+S_)
            R0_  = np.minimum(maxrate,sexp(LOGX))*dtfine
            RM = R0_ * np.minimum(1+0.5*LOGV,maxvcorr)
            J_   = Cb[None,:,:]*R0_[:,None,None]+Adt[None,:,:]
            allM1 += np.matmul(Adt,allM1[:,:,:])
            JM2_   = np.matmul(J_,allM2)
            allM2 += JM2_ + JM2_.transpose((0,2,1))
            allM1 +=  C[None,:,:]*RM[:,None,None]
            allM2 += CC[None,:,:]*RM[:,None,None]
        # Parallel measurement update
        M2B_ = np.matmul(allM2,beta)
        LV = allM2.dot(beta[:,0]).dot(beta[:,0])
        LV = np.maximum(1e-12,LV)
        LM = allM1[:,:,0].dot(beta[:,0])
        LT = 1/LV
        KG = M2B_/(VR+LV)[:,None,None]
        allM2 -= np.matmul(KG,M2B_.transpose(0,2,1))
        allM1 += KG*(MR-LM)[:,None,None]
        # Compute univariate update for likelihood
        TR  = 1/np.maximum(1e-7,VR)
        TP  = LT + TR
        VP  = 1/np.maximum(1e-7,TP)
        MP  = (LT*LM+TR*MR)*VP
        LOGR = MP+S_
        LOGPYX = Y_*LOGR-sexp(LOGR)
        LL = LOGPYX - 0.5*(slog(LV/VP) + (MP-LM)**2/LV)
    for i in range(N):
        M1 = allM1[i,:,0]
        M2 = allM2[i]
        allLR[i] = min(beta.T.dot(M1)+S[i],maxlogr)
        allLV[i] = beta.T.dot(M2).dot(beta)

    LL = LL[np.isfinite(LL)]
    return allLR,allLV,allM1,allM2,-np.nanmean(LL)
