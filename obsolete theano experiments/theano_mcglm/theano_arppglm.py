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
from   functions import sexp,slog
import warnings
from   warnings import warn

from scipy.linalg.matfuncs import expm

from scipy.linalg        import lstsq,pinv
from numpy.linalg.linalg import cholesky as chol
from numpy.linalg.linalg import LinAlgError

from measurements import *
from utilities    import *
from arguments    import *

COMPUTE_GRADIENTS = False

############################################################################
# Set up theano environment
dtype='float32'
import os
flags = 'mode=FAST_RUN,device=gpu,floatX=%s'%dtype
if dtype!='float64':
    flags += ',warn_float64=warn'
os.environ["THEANO_FLAGS"] = flags

import theano
import theano.tensor as T

############################################################################
# Set up an RNG
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

srng = RandomStreams(seed=234)

############################################################################
# Helper functions

def Tcon(x):
    return T.constant(x,dtype=dtype)

#eps     = np.finfo('float32').eps
eps     = np.sqrt(np.finfo('float32').tiny)
max_exp = Tcon(np.log(np.sqrt(np.finfo('float32').max)))

import theano.tensor.slinalg

Texpm = T.slinalg.Expm()
#Tmn   = T.minimum
#Tmx   = T.maximum

def Tmn(a,b):
    return Tcast(T.minimum(Tcast(a),Tcast(b)))

def Tmx(a,b):
    return Tcast(T.maximum(Tcast(a),Tcast(b)))

def Tcast(x):
    return T.cast(x,dtype)

TWOPI = Tcon(np.float128('6.283185307179586476925286766559005768394338798750211641949'))

def nozero(x):
    '''Clip number to be larger than `eps`'''
    #return Tcast(T.maximum(Tcast(eps),x))
    #return T.log(1+T.exp(x*10))/10
    return Tcast(Tmx(Tcast(eps),Tcast(x)))

def Tslog(x):
    '''Theano safe logarithm'''
    return Tcast(T.log(nozero(Tcast(x))))

def Tsexp(x):
    #x = T.minimum(max_exp,x)
    #x = Tcast(x)
    #return Tcast(T.exp(x))
    return Tcast(T.exp(Tmn(Tcast(max_exp),Tcast(x))))

def Tsinv(x):
    return Tcast(Tcon(1.0)/nozero(x))

def Tsdiv(a,x):
    return Tcast(Tcast(a)/nozero(x))

def Tfun(inp=None,out=None,upd=None):
    return theano.function(inputs               = inp,
                           outputs              = out,
                           updates              = upd,
                           on_unused_input      = 'warn',
                           allow_input_downcast = True)

def Tquickfun(theano_source,*args):
    print(args)
    return Tfun(inp = args, out=theano_source(*args))

from theano.compile.nanguardmode import NanGuardMode
NANGUARD = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)

############################################################################
# Sampling functions

def patch_moments(M1,M2):
    M1 = T.switch(T.isnan(M1), 0.0, M1)
    M2 = T.switch(T.isnan(M2), 0.0, M2)
    M1 = T.switch(T.isinf(M1), 0.0, M1)
    M2 = T.switch(T.isinf(M2), 0.0, M2)
    return M1,M2

def build_ML_GLM_likelihood_theano():
    '''
    This is the traditional loss used for 
    maxiumum likelihood PP-GLM estimation.
    '''

    # Input arguments
    Xall = T.matrix("Xall",dtype=dtype) # stimulus history features
    Ysp  = T.vector("Ysp",dtype=dtype) # spiking input
    par  = T.vector("par",dtype=dtype) # packed parameter vectors
    
    LR    = Tcast(Xall.dot(par[1:]) + par[0])
    R0    = Tcast(Tsexp(LR))
    EMNLL = Tcast(-T.mean(Ysp*LR-R0))
    
    # Filtering function
    f1 = Tfun(inp = [Xall,par],
              out = [LR])
  
    # Marginal expected likelihood with gradient
    f2 = Tfun(inp = [Xall,Ysp,par],
              out = [EMNLL])

    g2 = Tfun(inp = [Xall,Ysp,par], 
              out = [Tcast(theano.gradient.jacobian(Tcast(EMNLL),Tcast(par)))])

    h2 = Tfun(inp = [Xall,Ysp,par], 
              out = [theano.gradient.hessian(EMNLL,par)])
    return f1,f2,g2,h2

def get_integrator_theano(int_method,Adt):
    '''
    Generates either exponential-Euler or 
    linear forward-Euler integrators for moments
    '''
    F1   = Texpm(Adt)
    if int_method=="exponential":
        # Integrate locally linearized system via exponentiation
        mean_update = lambda M1: F1.dot(M1)
        def cov_update(M2,J):
            F2 = Texpm(J)
            return F2.dot(M2).dot(F2.T)
    elif int_method=="euler":
        # Integrate locally linearized system via forward Euler method
        mean_update = lambda M1: M1+Adt.dot(M1)
        def cov_update(M2,J):
            JM2 = J.dot(M2)
            return M2 + JM2 + JM2.T
    else:
        raise ValueError("int_method must be exponential or euler")
    return mean_update, cov_update

def get_update_theano(method,Cb,Adt,maxvcorr):
    # matrix
    # matrix
    # scalar
    
    '''
    Generate update function for moment integration.
    Valid methods are LNA, moment_closure, and second_order
    '''
    if method=="LNA":
        def update(logx,logv,Rm,M1,M2):
            # scalar scalar scalar column matrix
            return Rm,Cb*Rm+Adt
            # scalar matrix
    elif method=="second_order":
        def update(logx,logv,R0,M1,M2):
            # scalar scalar scalar column matrix
            Rm = R0 * Tmn(Tcast(1.0)+Tcast(0.5)*logv,maxvcorr)
            return Rm,Cb*R0+Adt
            # scalar matrix
    elif method=="moment_closure":
        def update(logx,logv,R0,M1,M2):
            # scalar scalar scalar column matrix
            Rm = R0 * Tmn(sexp(0.5*logv),maxvcorr)
            return Rm,Cb*Rm+Adt
            # scalar matrix
    elif method=="approximate_gamma":
        raise ValueError("approximate_gamma not implemented in Theano yet.")
        '''
        def update(logx,logv,R0,M1,M2):
            correction = Tmn(robust_expgammameancorrection(logv),maxvcorr)
            Rm = R0*correction
            return Rm,Cb*R0+Adt
        '''
    else:
        raise ValueError("method must be LNA, second_order, or moment_closure.\n"\
                         "`approximate_gamma` is defined, but is experimental.")
    return update

def project_moments(M1,M2,s,b,mxl):
    lv   = b.dot(M2).dot(b)       # scalar
    lv   = Tmx(Tcast(1e-9),lv)
    lx   = b.dot(M1[:,0])+s       # scalar 
    lx   = Tmn(mxl,lx)            # scalar
    #lx   = Tmx(-200,lx)
    return lx,lv

def build_integrate_moments_theano(N,A,C,
    dt         = 1.0,
    oversample = 10,
    maxrate    = 10,
    maxvcorr   = 100,
    method     = "moment_closure",
    int_method = "euler"):
    
    # Input arguments
    Xst  = T.matrix("Xst",dtype=dtype) # stimulus history features
    Ysp  = T.vector("Ysp",dtype=dtype) # spiking input
    par  = T.vector("par",dtype=dtype) # packed parameter vectors
    
    # Constant
    dtf   = Tcast(dt/oversample)
    Aop   = Tcon(A)
    Adtf  = Tcon(A*dt/oversample)
    F1    = Tcon(expm(A*dt/oversample))
    Cop   = Tcon(C)
    mxr   = Tcon(maxrate)
    mxvc  = Tcon(maxvcorr)
    mxl   = Tcon(slog(maxrate))
    K     = A.shape[0]
    
    # Unpack parameter vector
    b    = par[1:K+1] # spike history weights
    bst  = par[K+1:]  # stimulus weights
    mm   = par[0]     # constant offset
    
    # Pre-compute projected stimulus vector
    stim = mm + Xst.dot(bst)
    
    # Precompute some useful constants
    beta = b.dimshuffle(['x',0]) # Row 1xK
    Cop  = Tcon(C)          # Column Kx1
    Cb   = Cop.dot(beta)    # Matrix KxK
    CC   = Tcon(C.dot(C.T)) # Matrix KxK

    # Buid moment integrator functions
    mean_update, cov_update = get_integrator_theano(int_method,Adtf)
    
    # Get update function (computes expected rate from moments)
    update = get_update_theano(method,Cb,Adtf,mxvc)

    # Initial condition for moments
    M1 = Tcon(np.zeros((K,1)))
    M2 = Tcon(np.eye(K)*1e-6)

    def momentfilter(s,M1,M2):            # scalar, column, matrix
        M1,M2 = patch_moments(M1,M2)
        for j in range(oversample):
            lx,lv = project_moments(M1,M2,s,b,mxl)
            R0   = Tsexp(lx)              # scalar
            R0   = Tmn(mxr,R0)            # scalar
            R0  *= dtf                    # scalar
            Rm,J = update(lx,lv,R0,M1,M2) # scalar, matrix
            Rm   = Tmn(mxr,Rm)            # salar
            M2   = cov_update(M2,J)+CC*Rm # matrix
            M1   = mean_update(M1)+Cop*Rm # column
        lx,lv = project_moments(M1,M2,s,b,mxl)
        return lx,lv,M1,M2
    
    # Scan to compute moments, likelihood, measurement approximations
    [allLR,allLV,allM1,allM2], up = theano.scan(momentfilter,
                                    sequences     = [stim],
                                    outputs_info  = [None,None,M1,M2],
                                    non_sequences = [],
                                    n_steps       = N,
                                    name          = 'momentfilter')
    allLV = Tmx(0,allLV);
    
    # Filtering function
    f1 = Tfun(inp = [Xst,par],
              out = [allLR,allLV,allM1,allM2],
              upd = up)
    # Marginal expected likelihood with gradient
    R0   =  Tsexp(allLR)
    R1   =  R0*(1.0+0.5*allLV)
    ENLL = -T.mean(Ysp*allLR - R1)
    f2 = Tfun(inp = [Xst,Ysp,par],
              out = [ENLL],
              upd = up)
    if COMPUTE_GRADIENTS:
        g2 = Tfun(inp = [Xst,Ysp,par], 
                  out = [theano.gradient.jacobian(ENLL,par)],
                  upd = up)
    else:
        g2 = None
    return f1,f2,None

# Integration range
irange = Tcon(np.linspace(-4,4,25)) # vector
#irange = Tcon(np.linspace(-50,10,50)) # vector

def intmoment(m,v,y,s,dt,mxl,
    eps=1e-12): # scalar scalar scalar scalar scalar scalar 
    m0,s0   = m,T.sqrt(v)                       # scalar scalar
    x       = irange*s0 + m0                    # vector
    #x = irange
    logPx   = -0.5*(Tsdiv((x-m)**2,v)+Tslog(v)) # vector
    lograte = x+s+Tslog(dt)                     # vector
    lograte = Tmn(mxl,lograte)                  # vector
    #lograte = Tmx(-100,lograte)                  # vector
    logPyx  = y*lograte-Tsexp(lograte)          # vector
    logPyx -= T.max(logPyx)                     # vector
    Pxy     = Tsexp(logPyx+logPx)               # vector
    Pxy     = nozero(Pxy)                       # vector
    norm    = T.sum(Pxy)                        # scalar
    m       = T.sum(x*Pxy)/norm                 # scalar
    v       = T.sum((x-m)**2*Pxy)/norm          # scalar
    return m,v                                  # scalar scalar

def get_measurement_theano(measurement):
    '''
    Get measurement update function for forward filtering
    '''
    if measurement=='laplace':
        raise ValueError('Laplace update not implemented in Theano yet')
    if measurement=='variational':
        raise ValueError('variational update not implemented in Theano yet')
    if measurement in ('laplace','variational','moment'):
        measurement = {
            #'laplace'    :univariate_lgp_update_laplace,
            #'variational':univariate_lgp_update_variational,
            'moment'     :intmoment,
        }[measurement]
    elif not hasattr(measurement, '__call__'):
        raise ValueError("measurement must be variational, laplace, moment, or a function")
    return measurement

def condMVG_update_theano(M1,M2,y,b,beta,s,dt,rm,rp,mxl,measure,
    eps=Tcast(1e-12)):
    # column matrix scalar vector scalar scalar scalar scalar function
    # project moments
    m     = b.dot(M1[:,0])        # scalar
    v     = b.dot(M2).dot(b)      # scalar
    v     = Tmx(v,eps)            # scalar
    t     = Tsinv(v)              # scalar
    # Regularizing Gaussian prior on log-rate
    tq    = rp + t
    vq    = Tsinv(tq)
    mq    = (m*t+rm*rp)*vq
    # univariate measurement
    # --> scalar scalar
    mp,vp = measure(mq,vq,y,s,dt,mxl) 
    vp    = Tmx(vp,eps)           # scalar
    tp    = Tsinv(vp)             # scalar
    # surrogate measurement
    tr    = Tmx(tp-t,eps)         # scalar
    vr    = Tsinv(tr)             # scalar
    mr    = (mp*tp-m*t)*vr        # scalar
    # Conditional MVG update
    M2b   = M2.dot(beta.T)        # column
    K     = Tsdiv(M2b,vr+v)       # column
    M2    = M2-K.dot(M2b.T)       # matrix
    M1    = M1+K*(mr-m)           # column 
    # likelihood P(y) ~ P(y|x)*P(x)/P(x|y)
    lr    = Tmn(mxl,m+s)          # 1 vector
    lpyx  = (y*lr-Tsexp(lr))
    lpx   = -0.5*Tslog(v)
    lpxy  = -0.5*(Tslog(vp)+Tsdiv((m-mp)**2,vp))
    ll    = (lpyx+lpx-lpxy)
    return M1,M2,ll,mr,vr         # column matrix scalar scalar scalar

def surrogate_update_theano(M1,M2,y,b,beta,s,dt,rm,rp,mxl,measure,mr,vr,
    eps=Tcast(1e-5)):
    v      = Tmx(eps,b.dot(M2).dot(b))
    m      = b.dot(M1[:,0])
    M2b    = M2.dot(beta.T)
    K      = Tsdiv(M2b,vr+v) # K x 1
    M2     = M2 - K.dot(M2b.T) # (K x 1) (1 x K)
    M1     = M1 + K*(mr-m) 
    # univariate posterior for likelihood
    t      = Tsinv(v)
    tr     = Tsinv(vr)
    tp     = t + tr
    vp     = Tsinv(tp)
    mp     = (mr*tr+m*t)*vp
    logr   = mp+s
    logr   = Tmn(logr,mxl)
    logPyx = y*logr-Tsexp(logr)
    ll     = logPyx + 0.5*Tslog(vp/v) - 0.5*(mp-m)**2/v 
    return M1, M2, ll#scalar(ll)
                  
def build_filter_moments_theano(N,A,C,
    dt         = 1.0,
    oversample = 10,
    maxrate    = 10,
    maxvcorr   = 100,
    method     = "moment_closure",
    int_method = "euler",
    measurement = "moment",
    reg_cov     = 0.01,
    reg_rate    = 0.001,
    return_surrogates = False,
    use_surrogates    = False):
    
    # Input arguments
    Xst = T.matrix("Xst",dtype=dtype) # stimulus history features
    Ysp = T.vector("Ysp",dtype=dtype) # spiking input
    par = T.vector("par",dtype=dtype) # packed parameter vectors
    MR  = T.vector("MR",dtype=dtype)  # surrogate means
    VR  = T.vector("VR",dtype=dtype)  # surrogate variances
    
    # Constant
    dtf   = Tcast(dt/oversample)
    Aop   = Tcon(A)
    Adtf  = Tcon(A*dt/oversample)
    F1    = Tcon(expm(A*dt/oversample))
    Cop   = Tcon(C)
    mxr   = Tcon(maxrate)
    mxvc  = Tcon(maxvcorr)
    mxl   = Tcon(slog(maxrate))
    K     = A.shape[0]
    rcv   = Tcon(reg_cov)
    rr    = Tcon(reg_rate)
    llsc  = Tcon(1.0/N)
    
    # Unpack parameter vector
    mm   = par[0]     # constant offset
    b    = par[1:K+1] # spike history weights
    bst  = par[K+1:]  # stimulus weights
    
    # Pre-compute projected stimulus vector
    stim = mm + Xst.dot(bst)
    
    # Precompute some useful constants
    beta = b.dimshuffle(['x',0]) # Row 1xK
    Cop  = Tcon(C)               # Column Kx1
    Cb   = Cop.dot(beta)         # Matrix KxK
    CC   = Tcon(C.dot(C.T))      # Matrix KxK
    I    = T.eye(K,dtype=dtype)

    # Get measurement update function
    measurement = get_measurement_theano(measurement)
    # Buid moment integrator functions
    mean_update, cov_update = get_integrator_theano(int_method,Adtf)
    # Get update function (computes expected rate from moments)
    update = get_update_theano(method,Cb,Adtf,mxvc)

    # Initial condition for moments
    M1 = Tcon(np.zeros((K,1)))
    M2 = Tcast(np.eye(K,dtype=dtype)*1e-6)

    def integrate_dt(s,M1,M2,y):
        M1,M2 = patch_moments(M1,M2)
        if reg_cov>0:
            M2 = 0.5*(M2+M2.T) + rcv*I    # matrix 
        for j in range(oversample):
            lx,lv = project_moments(M1,M2,s,b,mxl)
            R0   = Tsexp(lx)              # scalar
            R0   = Tmn(mxr,R0)            # scalar
            R0  *= dtf                    # scalar
            Rm,J = update(lx,lv,R0,M1,M2) # scalar, matrix
            Rm   = Tmn(mxr,Rm)            # salar
            M2   = cov_update(M2,J)+CC*Rm # matrix
            M1   = mean_update(M1)+Cop*Rm # column
        ll = y*lx - Rm
        return M1,M2,ll
        
    
    if not use_surrogates:
        ####################################################################
        # Filter using approximations to non-conjugate update
        
        def momentfilter(s,y,M1,M2,nll):      # scalar scalar column matrix scalar
            M1,M2,ll = integrate_dt(s,M1,M2,y)
            M1,M2 = patch_moments(M1,M2)
            M1,M2,_,mr,vr = condMVG_update_theano(\
                     M1,M2,y,b,beta,s,dt,mm,rr,mxl,measurement)
            M1,M2 = patch_moments(M1,M2)
            lx,lv = project_moments(M1,M2,s,b,mxl)
            ll    = T.switch(T.isnan(ll),-1e3,ll)
            ll    = T.switch(T.isinf(ll),-1e3,ll)
            return lx,lv,M1,M2,nll-ll*llsc,mr,vr
        
        # Scan to compute moments, likelihood, measurement approximations
        [allLR,allLV,allM1,allM2,cnll,mr,vr], up = theano.scan(momentfilter,
                                        sequences     = [stim,Ysp],
                                        outputs_info  = [None,None,M1,M2,Tcon(0),None,None],
                                        non_sequences = [],
                                        n_steps       = N,
                                        name          = 'momentfilter')
        
        f1 = Tfun(inp = [Xst,Ysp,par],
                  out = [allLR,Tmx(0,allLV),allM1,allM2,cnll[-1]]\
                      +([mr,vr] if return_surrogates else []),
                  upd = up)
    else:
        ####################################################################
        # Surrogate gaussian measurements provided
        
        def momentfilter(s,y,mr,vr,M1,M2,nll):      # scalar scalar column matrix scalar
            M1,M2,ll = integrate_dt(s,M1,M2,y)
            M1,M2 = patch_moments(M1,M2)
            M1,M2,_ = surrogate_update_theano(\
                     M1,M2,y,b,beta,s,dt,mm,rr,mxl,measurement,mr,vr)
            M1,M2 = patch_moments(M1,M2)
            lx,lv = project_moments(M1,M2,s,b,mxl)
            ll    = T.switch(T.isnan(ll),-1e3,ll)
            ll    = T.switch(T.isinf(ll),-1e3,ll)
            return lx,lv,M1,M2,nll-ll*llsc
        
        # Scan to compute moments, likelihood, measurement approximations
        [allLR,allLV,allM1,allM2,cnll], up = theano.scan(momentfilter,
                                        sequences     = [stim,Ysp,MR,VR],
                                        outputs_info  = [None,None,M1,M2,Tcon(0)],
                                        non_sequences = [],
                                        n_steps       = N,
                                        name          = 'momentfilter')
        f1 = Tfun(inp = [Xst,Ysp,par,MR,VR],
                  out = [allLR,Tmx(0,allLV),allM1,allM2,cnll[-1]],
                  upd = up)
                    
    # Negative log-likelihood and gradient
    f2 = Tfun(inp = [Xst,Ysp,par]+([MR,VR] if use_surrogates else []),
              out = [cnll[-1]],
              upd = up)
    if COMPUTE_GRADIENTS:
        g2 = Tfun(inp = [Xst,Ysp,par]+([MR,VR] if use_surrogates else []), 
                  out = [theano.gradient.jacobian(cnll[-1],par)],
                  upd = up)
    else:
        g2 = None
    return f1,f2,g2
    
def build_parallel_shallow_moments(N,A,C,depth,
    dt         = 1.0,
    oversample = 10,
    maxrate    = 10,
    maxvcorr   = 100,
    method     = "moment_closure",
    int_method = "euler",
    measurement = "moment",
    reg_cov     = 0.01,
    reg_rate    = 0.001,
    return_surrogates = False,
    use_surrogates    = False):
    '''
    Filter a fixed depth from ALL time-points
    '''
    if not depth<N:
        raise ValueError('Depth should be less than sequence length')

    # Number of time points that can be filtered to depth        
    ND = N-depth+1
      
    # Input arguments
    Xst = T.matrix("Xst",dtype=dtype) # stimulus history features
    Ysp = T.vector("Ysp",dtype=dtype) # spiking input
    par = T.vector("par",dtype=dtype) # packed parameter vectors
    MR  = T.vector("MR",dtype=dtype)  # surrogate means
    VR  = T.vector("VR",dtype=dtype)  # surrogate variances
    
    # Constant
    dtf   = dt/oversample
    Aop   = Tcon(A)
    Adtf  = Tcon(A*dtf)
    F1    = Tcon(expm(A*dtf))
    Cop   = Tcon(C)
    mxr   = Tcon(maxrate)
    mxvc  = Tcon(maxvcorr)
    mxl   = Tcon(slog(maxrate))
    K     = A.shape[0]
    rcv   = Tcon(reg_cov)
    rr    = Tcon(reg_rate)
    llsc  = Tcon(1.0/N)
    
    # Unpack parameter vector
    mm   = par[0]     # constant offset
    b    = par[1:K+1] # spike history weights
    bst  = par[K+1:]  # stimulus weights
    
    # Pre-compute projected stimulus vector
    stim = mm + Xst.dot(bst)
    
    # Precompute some useful constants
    beta = b.dimshuffle(['x',0]) # Row 1xK
    Cop  = Tcon(C)               # Column Kx1
    Cb   = Cop.dot(beta)         # Matrix KxK
    CC   = Tcon(C.dot(C.T))      # Matrix KxK
    I    = T.eye(K,dtype=dtype)

    # Get measurement update function
    measurement = get_measurement_theano(measurement)
    # Buid moment integrator functions
    mean_update, cov_update = get_integrator_theano(int_method,Adtf)
    # Get update function (computes expected rate from moments)
    update = get_update_theano(method,Cb,Adtf,mxvc)

    # Initial condition for moments
    # We DO NOT use scan for this!
    
    mat2all = T.ones(ND).dimshuffle([0,'x','x'])
    vec2all = T.ones(ND).dimshuffle([0,'x'])
    
    allM1 = M1*vec2all
    allM2 = Mc*mat2all

    # TODO: modify
    # this needs to operate over many values in parallel
    def integrate_dt(s,allM1,allM2):
        if reg_cov>0:
            M2 = 0.5*(M2+M2.T) + rcv*I    # matrix 
        for j in range(oversample):
            lv   = b.dot(M2).dot(b)       # scalar
            lx   = b.dot(M1[:,0])+s       # scalar 
            lx   = Tmn(mxl,lx)            # scalar
            R0   = Tsexp(lx)              # scalar
            R0   = Tmn(mxr,R0)            # scalar
            R0  *= dtf                    # scalar
            Rm,J = update(lx,lv,R0,M1,M2) # scalar, matrix
            M2   = cov_update(M2,J)+CC*Rm # matrix
            M1   = mean_update(M1)+Cop*Rm # column
        return M1, M2
    
    for i in range(depth):
        allM1,allM2 = integrate_dt(allS[i],allM1,allM2)
        # Then do parallel surrogate measurement update
        # this is not trivial

        # This will take a couple days to figure
        # implement in numpy first
        # not now
        
#
def Tmatmul(a: T.TensorType, b: T.TensorType, _left=False):
    """Replicates the functionality of numpy.matmul, except that
    the two tensors must have the same number of dimensions, and their ndim must exceed 1."""

    # TODO ensure that broadcastability is maintained if both a and b are broadcastable on a dim.

    assert a.ndim == b.ndim  # TODO support broadcasting for differing ndims.
    ndim = a.ndim
    assert ndim >= 2

    # If we should left multiply, just swap references.
    if _left:
        tmp = a
        a = b
        b = tmp

    # If a and b are 2 dimensional, compute their matrix product.
    if ndim == 2:
        return T.dot(a, b)
    # If they are larger...
    else:
        # If a is broadcastable but b is not.
        if a.broadcastable[0] and not b.broadcastable[0]:
            # Scan b, but hold a steady.
            # Because b will be passed in as a, we need to left multiply to maintain
            #  matrix orientation.
            output, _ = theano.scan(Tmatmul, sequences=[b], non_sequences=[a[0], 1])
        # If b is broadcastable but a is not.
        elif b.broadcastable[0] and not a.broadcastable[0]:
            # Scan a, but hold b steady.
            output, _ = theano.scan(Tmatmul, sequences=[a], non_sequences=[b[0]])
        # If neither dimension is broadcastable or they both are.
        else:
            # Scan through the sequences, assuming the shape for this dimension is equal.
            output, _ = theano.scan(Tmatmul, sequences=[a, b])
        return output
