#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import sys
import numpy as np
import scipy
import warnings

from scipy.linalg        import lstsq,pinv
from numpy.linalg.linalg import cholesky as chol
from numpy.linalg.linalg import LinAlgError

from measurements import *
from utilities    import *
from arguments    import *
from functions    import sexp,slog
from utilities import current_milli_time

def ensemble_sample(stim,B,beta,M=100,maxrate=100):
    '''
    Sample from an autoregressive point-process model. 
    
    This is discrete time, and allows only one spike per time point.
    It may under-estimate self-excitation. 
    
    Parameters
    ----------
    stim: Effective input. Incorporates filtered stimulus and constant offset
    B: History basisa
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
    maxlograte = np.log(maxrate)
    for i,s in enumerate(stim):
        l[i,:]  = s + b.dot(h)
        rate    = np.exp(np.minimum(maxlograte,l[i]))
        y[i,:]  = np.random.poisson(rate)>0
        h[1:,:] = h[:-1,:]
        h[0,:]  = y[i,:]
    return y,l

def ensemble_sample_continuous(stim,B,beta,oversample=10,M=100,maxrate=100):
    '''
    Sample from an autoregressive point-process model. 
    Use continuous time
    
    Parameters
    ----------
    stim: Effective input. Incorporates filtered stimulus and constant offset
    B: History basisa
    beta: Projection from the history basis onto log-rate

    Other Parameters
    ----------------
    M: number of samples to draw
    
    Returns
    -------
    y : ntime x nsamples matrix of point-process samples
    '''
    K,N = B.shape
    
    # Create discrete differentiation operator
    Dtau = -np.eye(N) + np.eye(N,k=-1)
    # Create delta operator (to inject signal into delay line)
    S = np.zeros((N,1))
    S[0,0] = 1
    # Perform a change of basis from function space into the basis projection B
    A = B.dot(Dtau).dot(pinv(B))
    C = B.dot(S)
    
    T = len(stim)
    z = np.zeros((K,M))
    dtf = 1.0 / oversample
    F1  = scipy.linalg.expm(A*dtf)
    ally = np.zeros((T,M))
    alll = np.zeros((T,M))
    maxlograte = np.log(maxrate)
    time = 0.0
    for i,s in enumerate(stim):
        ytotal = np.zeros(M)
        while time<i+1:
            l  = s + beta.T.dot(z)
            dt = np.exp(-(np.max(l)+1))
            dt = min(0.2,dt)
            dt = max(1e-12,dt)
            lr = np.minimum(l+np.log(dt),maxlograte)
            r  = np.exp(lr)
            y  = np.random.poisson(r).ravel()>1
            z  = scipy.linalg.expm(A*dt).dot(z) + C*y
            time += dt
            ytotal += y
        alll[i,:] = s + beta.T.dot(z)
        ally[i,:] = ytotal
    return ally,alll

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
    
    
def langevin_sample_history(stim,A,beta,C,
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
    '''
    maxlograte = np.log(maxrate)
    dtfine     = dt/oversample
    T   = len(stim)
    K   = beta.size
    Adt = A*dtfine
    z   = np.zeros((K,M))
    Z   = np.zeros((T,K,M))
    l   = np.zeros((T,M))
    for i,s in enumerate(stim):
        for j in range(oversample):
            lr = np.minimum(s + beta.T.dot(z),maxlograte)
            r = np.exp(lr)*dtfine
            Y = np.random.randn(M)*np.sqrt(r) + r
            z = z + Adt.dot(z) + C*Y
        Z[i] = z
        l[i] = lr
    return Z,l

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
    '''
    Sample the log rate and basis-approximated history process from the
    Langevin approximation.
    '''
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
    
    We have not investigated the overall accuracy of this alternative 
    moment-closure. 
    
    This moment closure appears to suffer from large errors when variance
    is large, possibly similar to the effect of outliers observed in the 
    log Gaussian moment closure. 
    
    However, it is possible to use a polynomial approximation to this 
    moment closure that is accurate for "reasonable" values of the variance,
    and diverges less dramatically for large variance. Please see the
    function `robust_expgammameancorrection`.
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
    
def robust_expgammameancorrection(v):
    '''
    Quadratic regression for gamma moment-matching for small variance.
    This may be more accurate in some cases but in general has not been
    evaluated for accuracy, it could be worse in others.
    
    This was obtained by regressing a quadratic polynomial to the 
    exact solution for gamma moment-matching of the mean / variance for
    predicting <λ>.
    '''
    return 1+0.33257734*v+0.04030693*v**2

def get_update_function(method,Cb,Adt,maxvcorr):
    '''
    Generate update function for moment integration.
    
    Valid methods are LNA, moment_closure, and second_order

    Parameters
    ----------
    method: string
        Valid methods are `LNA`, `moment_closure`, and `second_order`
    Cb: matrix
        Precomputed C.dot(beta.T), where C is the basis-projected 
        "signal-injection" operator for the history process
    Adt: square matrix
        forward evolution operator premultiplied by time step
    maxvcorr: scalar
        Maximum corrected variance (for artifically enforcing hard-
        limits on stability of the ODE moment equations)

    Returns
    -------
    update: function
        Update function accepting parameters `logx`, `logv`, `R0`, `M1`, and `M2`,
        where `logx` and `logv` are the means and the variance of the log-intensity,
        `R0` is the uncorrected estimate of the mean intensity. Function returns
        the corrected mean `Rm` and the system Jacobian `J`.
    '''
    if method=="LNA":
        def update(logx,logv,Rm,M1,M2):
            J = Cb*Rm+Adt
            return Rm,J
    elif method=="second_order":
        def update(logx,logv,R0,M1,M2):
            Rm = R0 * min(1+0.5*logv,maxvcorr)
            J  = Cb*R0+Adt
            return Rm,J
    elif method=="moment_closure":
        def update(logx,logv,R0,M1,M2):
            Rm = R0 * min(sexp(0.5*logv),maxvcorr)
            J  = Cb*Rm+Adt
            return Rm,J
    elif method=="approximate_gamma":
        def update(logx,logv,R0,M1,M2):
            correction = min(robust_expgammameancorrection(logv),maxvcorr)
            Rm = R0*correction
            J  = Cb*R0+Adt
            return Rm,J
    else:
        raise ValueError("method must be LNA, second_order, or moment_closure.\n"\
                         "`approximate_gamma` is defined, but is experimental.")
    return update

def get_moment_integrator(int_method,Adt):
    '''
    Generates either exponential-Euler or 
    linear forward-Euler integrators for moments

    Parameters
    ----------
    int_method: string
        One of either "exponential", "approximate_exponential", or "euler"
    Adt: square matrix
        forward evolution operator premultiplied by time step

    Returns
    -------
    mean_update: function: vector → vector
        Mean update function, updates mean vector one time step
    cov_update: function: matrix → matrix
        Covariance update function, updates covariance matrix one time step.
    '''
    F1  = scipy.linalg.expm(Adt)
    I   = np.eye(Adt.shape[0])
    if int_method=="exponential":
        # Integrate locally linearized system via exponentiation
        mean_update = lambda M1: F1.dot(M1)
        def cov_update(M2,J):
            F2 = scipy.linalg.expm(J)
            return F2.dot(M2).dot(F2.T)
    elif int_method=="approximate_exponential":
        # Exponential for the means, but use 1st-order Taylor for
        # getting the exponential for the covariance
        mean_update = lambda M1: F1.dot(M1)
        def cov_update(M2,J):
            F2 = I + J
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

def integrate_moments(stim,A,beta,C,
    dt         = 1.0,
    oversample = 10,
    maxrate    = 500,
    maxvcorr   = 2000,
    method     = "moment_closure",
    int_method = "euler",
    reg_cov    = 1e-6,
    safemode   = "assert"):
    '''
    Integrate moment equations for autoregressive PPGLM
    
    Parameters
    ----------
    stim : zero-lage effective input (filtered stimulus plus mean offset)
    A : forward operator for delay-line evolution
    beta : basis history weights
    C : projection of current state onto delay-like
    
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
    reg_cov: 
        Small diagonal regularization for covariance matrix
    safemode: string
        Whether to check, repair NaNs and singular covariance. 
        if "assert", NaNs will trigger assertion error,
        if "repair", NaNs will be silently corrected
    
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
    # Buid moment integrator functions
    mean_update, cov_update = get_moment_integrator(int_method,Adt)
    # Get update function (computes expected rate from moments)
    update = get_update_function(method,Cb,Adt,maxvcorr)
    # Initial condition for moments
    M1 = np.zeros((K,1))
    M2 = np.eye(K)*1e-6
    # Store moments
    allM1 = np.zeros((T,K))
    allM2 = np.zeros((T,K,K))
    allLR = np.zeros((T))
    allLV = np.zeros((T))
    # Integrate
    for i,s in enumerate(stim):
        for j in range(oversample):
            if   safemode=="assert":
                assert(np.all(np.isfinite(M1)) and np.all(np.isfinite(M2)))
            elif safemode=="repair":
                M1[~np.isfinite(M1)] = 0
                M2[~np.isfinite(M2)] = 0
            # Marginal variance and mean of the predicted log-rate
            # (This is a linear projection of the process history)
            logv  = beta.T.dot(M2).dot(beta)
            logx  = min(beta.T.dot(M1)+s,maxlogr)
            # Uncorrected rate estimate
            R0    = sexp(logx)*dtfine
            # Covariance corrected rate estimate Rm
            # And system Jacobian J
            Rm,J  = update(logx,logv,R0,M1,M2)
            # Moment updates with new mean and variance
            # proportional to corrected rate Rm
            M2    = cov_update(M2,J) + CC*Rm
            M1    = mean_update(M1)  + C *Rm
        if safemode=="repair":
            M2 = repair_covariance(M2,reg_cov)
        allM1[i] = M1[:,0]
        allM2[i] = M2
        allLR[i] = logx
        allLV[i] = beta.T.dot(M2).dot(beta)
    return allLR,allLV,allM1,allM2

def get_measurement(measurement):
    '''
    Get measurement update function for forward filtering
    '''
    if measurement in ('laplace','variational','moment','variational_so'):
        measurement = {
            'laplace'       :univariate_lgp_update_laplace,
            'variational'   :univariate_lgp_update_variational,
            'moment'        :univariate_lgp_update_moment,
            'variational_so':univariate_lgp_update_variational_so, # experimental
        }[measurement]
    elif not hasattr(measurement, '__call__'):
        raise ValueError("measurement must be variational, laplace, moment, or a function")
    return measurement

def get_ll(M1,C1,M2,C2,beta,s,y):
    '''
    Compute negative log-likelihood up to a constant.

    Parameters
    ----------
    M1: prior mean
    C1: prior covariance
    M2: posterior mean
    C2: posterior covariance
    beta: projection onto rate
    s: offset 
    y: point-process observation count
    '''
    M1   = assertfinitereal(ascolumn(M1))
    M2   = assertfinitereal(ascolumn(M2))
    C1   = assertfinitereal(assquare(C1))
    C2   = assertfinitereal(assquare(C2))
    beta = assertfinitereal(ascolumn(beta))
    logr   = beta.T.dot(M1)+s
    logPyx = y*logr - sexp(logr)
    Ch     = trychol(C1,1e-6)
    RR     = linv(Ch,M2-M1)
    ll     = logPyx + np.sum(slog(np.diag(chol(C2)))) - np.sum(slog(np.diag(Ch))) - 0.5*RR.T.dot(RR)
    return scalar(ll)

def get_ll_univariate(m1,v1,m2,v2,beta,s,y):
    '''
    Compute negative log-likelihood up to a constant

    Parameters
    ----------
    m1: prior mean
    v1: prior covariance
    m2: posterior mean
    v2: posterior covariance
    beta: projection onto rate
    s: offset 
    y: point-process observation count
    '''
    m1 = scalar(m1)
    v1 = scalar(v1)
    m2 = scalar(m2)
    v2 = scalar(v2)
    logr = m2+s
    logPyx = y*logr - sexp(logr)
    ll = logPyx + 0.5*slog(v2/v1) - 0.5*(m2-m1)**2/v1 
    return scalar(ll)

def filter_moments(stim,Y,A,beta,C,m,
    dt          = 1.0,
    oversample  = 10,
    maxrate     = 500,
    maxvcorr    = 2000,
    method      = "moment_closure",
    int_method  = "euler",
    measurement = "moment",
    reg_cov     = 0,
    reg_rate    = 0,
    return_surrogates  = False,
    use_surrogates     = None,
    initial_conditions = None,
    progress           = False,
    safe               = True):
    
    '''
    Parameters
    ----------
    stim : zero-lage effective input (filtered stimulus plus mean offset)
    Y : point-process count observations, same length as stim
    A : forward operator for delay-line evolution
    C : projection of current state onto delay-like
    beta : basis history weights
    m : log-rate bias parameter, log-rates are regularized toward this value
    
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
    measurement:
        "moment", "laplace", or "variational"
    reg_cov:
        Diagonal covariance regularization
    reg_rate:
        Small regularization toward log mean-rate; This parameter reflects
        the precision of a Gaussian prior about the log mean-rate, applied 
        at every measurement update.
    return_surrogates: bool
        If true, Gaussian approximations of measurement likelihoods are
        returned. 
    use_surrogates: None or tuple
        Can be set as tuple of (means, variances) for Gaussian
        approximations of meausrement updates.
    initial_conditions: None or tuple
        Can be set to a tuple (M1,M2) of initial conditions for moment
        filtering.
    progress: boolean
        Whether to report progress
    
    Returns
    -------
    allLR : single-time marginal mean of log-rate
    allLV : single-time marginal variance of log-rate
    allM1 : low-dimensional approximation of history process, mean
    allM2 : low-dimensional approximation of history process, covariance
    nll   : negative log-likelihood
    '''
    # check arguments
    stim = asvector(stim)
    Y    = asvector(Y)
    A    = assquare(A)
    if oversample<1:
        raise ValueError('oversample must be non-negative integer')
    if method=="moment_closure" and measurement=="variational":
        warnings.warn("There are unresolved numerical stability issues "\
        "when using the log-Gaussian variational update with Gaussian "\
        "moment closure. Suggest using the second-order moment closure "\
        "instead")
    # Precompute constants
    maxlogr   = np.log(maxrate)
    maxratemc = maxvcorr*maxrate
    dtfine    = dt/oversample
    T         = len(stim)
    K         = beta.size
    I         = np.eye(K)
    Cb        = C.dot(beta.T)
    CC        = C.dot(C.T)
    Adt       = A*dtfine
    if not use_surrogates is None:
        MR,VR = use_surrogates
    # Get measurement update function
    measurement = get_measurement(measurement)
    # Buid moment integrator functions
    mean_update, cov_update = get_moment_integrator(int_method,Adt)
    # Get update function (computes expected rate from moments)
    update = get_update_function(method,Cb,Adt,maxvcorr)
    # Accumulate negative log-likelihood up to a constant
    nll = 0
    llrescale = 1.0/len(stim)
    if initial_conditions is None:
        # Initial condition for moments
        M1 = np.zeros((K,1))
        M2 = np.eye(K)*1e-6
    else:
        M1,M2 = initial_conditions
    # Store moments
    allM1 = np.zeros((T,K))
    allM2 = np.zeros((T,K,K))
    allLR = np.zeros((T))
    allLV = np.zeros((T))
    allmr = np.zeros((T))
    allvr = np.zeros((T))
    if progress:
        last_shown = current_milli_time()
    for i,s in enumerate(stim):
        # Regularize
        if reg_cov>0:
            strength = reg_cov+max(0,-np.min(np.diag(M2)))
            M2 = 0.5*(M2+M2.T) + strength*np.eye(K) 
        # Integrate moments forward
        for j in range(oversample):
            logv  = beta.T.dot(M2).dot(beta)
            logx  = min(beta.T.dot(M1)+s,maxlogr)
            R0    = sexp(logx)
            R0    = min(maxrate,R0)
            R0   *= dtfine
            Rm,J  = update(logx,logv,R0,M1,M2)
            M2    = cov_update(M2,J) + CC*Rm
            M1    = mean_update(M1)  + C *Rm
            if safe:
                M1    = np.clip(M1,-100,100)
                M2    = np.clip(M2,-100,100)
        # Measurement update
        pM1,pM2 = M1,M2
        if use_surrogates is None:
            # Use specified approximation method to handle non-conjugate
            # log-Gaussian Poisson measurement update
            M1,M2,ll,mr,vr = measurement_update_projected_gaussian(\
                      M1,M2,Y[i],beta,s,dt,m,reg_rate,measurement)
            allmr[i] = mr
            allvr[i] = vr
        else:
            # Use specified Gaussian approximations (MR,VR) to the 
            # measurement likelihoods.
            M1,M2,ll = measurement_update_projected_gaussian_surrogate(\
                      M1,M2,Y[i],beta,s,dt,m,reg_rate,measurement,
                                      return_surrogate=False,
                                      surrogate=(MR[i],VR[i]))
        if safe:
            M1    = np.clip(M1,-100,100)
            M2    = np.clip(M2,-100,100)
        # Store moments
        allM1[i] = M1[:,0].copy()
        allM2[i] = M2.copy()
        allLR[i] = min(beta.T.dot(M1)+s,maxlogr)
        allLV[i] = beta.T.dot(M2).dot(beta)
        nll -= ll*llrescale
        if safe:
            # Heuristic: detect numerical failure and exit early
            failed = np.any(M1)<-1e5
            failed|= logx>100*maxlogr
            failed|= nll<-1e10
            if failed:
                nll = np.inf
                break
        if progress and current_milli_time()-last_shown>500:
            sys.stdout.write('\r%02.02f%%'%(i*100/T))
            sys.stdout.flush()
            last_shown = current_milli_time()
    if progress:
        sys.stdout.write('\r100.00%')
        sys.stdout.flush()
        sys.stdout.write('\n')
    if return_surrogates:
        return allLR,allLV,allM1,allM2,nll,allmr,allvr
    else:
        return allLR,allLV,allM1,allM2,nll
    
