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
import warnings
from scipy.linalg import lstsq,pinv
from numpy.linalg.linalg import cholesky as chol
from numpy.linalg.linalg import LinAlgError

from utilities import *
from arguments import *

import traceback

import sys
from utilities import current_milli_time

from plot import v2str_long

def minimize_retry(objective,initial,jac=None,hess=None,
                   verbose=False,
                   printerrors=True,
                   failthrough=True,
                   tol=1e-5,
                   simplex_only=False,
                   show_progress=True,
                   **kwargs):
    '''
    call `scipy.optimize.minimize`, retrying a few times in case
    one solver doesn't work
    
    # Try a few things before giving up
    # Newton-CG should be pretty fast, but Hessian might have numeric problems
    # If this failes, try the default BFGS solver, which uses only the gradient
    # If this failes, try the simplex algorithm
    
    There are some bugs in some optimization routines, 
    also try to catch that stuff
    '''
    result = None
    x0     = np.array(initial).ravel()
    g0     = 1/np.zeros(x0.shape)
    
    nfeval = 0
    ngeval = 0
    
    if jac is True:
        v,g = objective(x0)
        best   = v
    else:
        v = objective(x0)
        best   = v

    if show_progress:
        sys.stdout.write('\n')
        last_shown = current_milli_time()
    def progress_update():
        if show_progress: 
            nonlocal best, x0, nfeval, ngeval, last_shown
            if current_milli_time() - last_shown > 500:
                ss = np.float128(best).astype(str)
                ss += ' '*(20-len(ss))
                out = '\rNo. function evals %6d \tNo. grad evals %6d \tBest value %s'%(nfeval,ngeval,ss)
                sys.stdout.write(out)
                sys.stdout.flush()
                last_shown = current_milli_time()
    def clear_progress():
        if show_progress: 
            progress_update()
            sys.stdout.write('\n')
            sys.stdout.flush()

    if jac is True:
        def wrapped_objective(params):
            nonlocal best, x0, nfeval, ngeval
            v,g = objective(params)
            if np.isfinite(v) and v<best:
                best = v
                x0   = params
            nfeval += 1
            ngeval += 1
            progress_update()
            return v,g
    else:
        def wrapped_objective(params):
            nonlocal best, x0, nfeval
            v = objective(params)
            if np.isfinite(v) and v<best:
                best = v
                x0   = params
            nfeval += 1
            progress_update()
            return v 
    
    if hasattr(jac, '__call__'):
        # Jacobain is function
        original_jac = jac
        def wrapped_jacobian(params):
            nonlocal best, x0, nfeval, ngeval
            nonlocal best, x0
            g = original_jac(params)
            ngeval += 1
            progress_update()
            return g
        jac = wrapped_jacobian
    
    def try_to_optimize(method,validoptions,jac_=None):
        try:
            options = {k:v for (k,v) in kwargs.items() if k in validoptions.split()}
            result = scipy.optimize.minimize(wrapped_objective,x0.copy(),
                jac=jac_,hess=hess,method=method,tol=tol,**options)
            _ = wrapped_objective(result.x)
            clear_progress()
            if result.success: 
                return True
            if verbose or printerrors:
                sys.stderr.write('%s reported "%s"\n'%(method,result.message))
                sys.stderr.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            clear_progress()
            if verbose or printerrors:
                sys.stderr.write('Error using minimize with %s:\n'%method)
                sys.stderr.flush()
                traceback.print_exc()
                sys.stderr.flush()
        finally:
            return False
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message='Method Nelder-Mead does not use')
            warnings.filterwarnings("ignore",message='Method BFGS does not use')
            # If gradient is provided....
            if not jac is None and not jac is False and not simplex_only:
                if try_to_optimize('Newton-CG','disp xtol maxiter eps',jac_=jac):
                    return x0
                if try_to_optimize('BFGS','disp gtol maxiter eps norm',jac_=jac):
                    return x0
            # Without gradient...
            if not simplex_only:
                if try_to_optimize('BFGS','disp gtol maxiter eps norm',\
                    jac_=True if jac is True else None):
                    return x0
            # Simplex is last resort, slower but robust
            if try_to_optimize('Nelder-Mead',
                    'disp maxiter maxfev initial_simplex xatol fatol',
                    jac_=True if jac is True else None):
                return x0
    except (KeyboardInterrupt, SystemExit):
        #return x0
        print('Best parameters are %s with value %s'%(v2str_long(x0),best))
        raise

    if failthrough:
        if verbose:
            sys.stderr.write('Minimization may not have converged\n')
            sys.stderr.flush()
        return x0 # fail through
    raise ArithmeticError('All minimization attempts failed')


def univariate_lgp_update_moment(m,v,y,s,dt,
                    tol     = 1e-3,
                    maxiter = 100,
                    eps     = 1e-7):
    # Get moments by integrating
    v = max(v,eps)
    t = 1/v
    m = np.clip(m,-200,20)
    # Set integration limits
    m0,s0 = (m,np.sqrt(v)) if t>1e-6 else (slog(y+0.25),np.sqrt(1/(y+1)))
    x = np.linspace(m0-4*s0,m0+4*s0,25)
    # Calculate likelihood contribution
    rate = x + s + slog(dt)
    l = y*(rate)-sexp(rate)
    # Normalize to prevent overflow
    l-= np.max(l)
    # Calculate prior contribution
    l += -.5*(x-m)**2/v-.5*slog(v)
    # Estimate posterior
    p = sexp(l)
    p[p<eps] = eps   
    p /= np.sum(p)
    # Integrate to get posterior moments
    m  = np.sum(x*p)
    v  = np.sum((x-m)**2*p)
    assertfinitereal(m)
    assertfinitereal(v)
    return m,v
    
def univariate_lgp_update_laplace(m,v,y,s,dt,
                    tol     = 1e-6,
                    maxiter = 100,
                    eps     = 1e-12):
    '''
    Optimize using Laplace approximation
    '''
    v = max(v,eps)
    scale = dt*np.exp(s)
    def objective(mu):
        rate = scale*sexp(mu)
        if not np.isfinite(mu) or not np.isfinite(rate):
            return np.inf
        return -y*mu+rate+0.5*(mu-m)**2/v
    def gradient(mu):
        rate = scale*sexp(mu)
        if not np.isfinite(mu) or not np.isfinite(rate):
            return np.inf
        return -y+rate+(mu-m)/v
    def hessian(mu):
        rate = scale*sexp(mu)
        if not np.isfinite(mu) or not np.isfinite(rate):
            return np.inf
        return rate+1/v
    # Try a few things before giving up
    # Newton-CG should be pretty fast, but Hessian might have numeric problems
    # If this failes, try the default BFGS solver, which uses only the gradient
    # If this failes, try the simplex algorithm
    mu = minimize_retry(objective,m,gradient,hessian,tol=1e-6,show_progress=False,printerrors=False)
    vv = 1/hessian(mu)
    return mu, vv

def univariate_lgp_update_variational(m,v,y,s,dt,
                    tol     = 1e-6,
                    maxiter = 100,
                    eps     = 1e-12):
    '''
    Optimize variational approximation
    Mean and variance must be optimized jointly
    '''
    v = max(v,eps)
    scale = dt*sexp(s)
    # Use minimization to solve for variational solution
    def objective(parameters):
        mq,vq = parameters
        rate  = scale*np.exp(mq + 0.5*vq)
        if vq<eps or not np.isfinite(mq) or not np.isfinite(vq) or not np.isfinite(rate):
            return np.inf
        return -y*mq + rate + 0.5*( -slog(vq) + (vq + (mq-m)**2)/v )
    def gradient(parameters):
        mq,vq = parameters
        rate  = scale*np.exp(mq + 0.5*vq)
        if vq<eps or not np.isfinite(mq) or not np.isfinite(vq) or not np.isfinite(rate):
            return (np.NaN,np.NaN)
        dm    = -y + rate + (mq-m)/v
        dv    = rate*0.5 + 0.5*(1/v-1/vq)
        return np.array([dm, dv]).squeeze()
    def hessian(parameters):
        mq,vq = parameters
        rate  = scale*np.exp(mq + 0.5*vq)
        if vq<eps or not np.isfinite(mq) or not np.isfinite(vq) or not np.isfinite(rate): 
            return [[np.NaN,np.NaN],[np.NaN,np.NaN]]
        dmdm  = rate + 1/v
        dvdm  = rate*0.5
        dvdv  = rate*0.25 + 0.5/vq**2
        return np.array([[dmdm,dvdm],[dvdm,dvdv]]).squeeze()
    return minimize_retry(objective,[m,v],gradient,hessian,tol=1e-6,show_progress=False,printerrors=False)

    
def assertshape(M,shape):
    if isinstance(M,(tuple,list)):
        M = np.array(M)
    if not M.shape==shape:
        raise ValueError('Expected shape (%s) but found shape (%s)'%(shape,M.shape))
    return M

def measurement_update_projected_gaussian(m1,m2,y,b,s,dt,pm,pt,
                                          univariate_method = univariate_lgp_update_moment,
                                          eps=1e-7,
                                          safe=False):
    """
    
    
    Derivation of Kalman update stored here
    
    
    '''
    # Original multivariate update (slower)
    # Multivariate update
    #bt  = b*tr
    #c   = I+m2.dot(bt.dot(b.T))
    #m1p = ascolumn(linv(c,m2.dot(bt*mr) + m1))
    #m2p = linv(c,m2)
    '''
    '''
    # Perform Multivariate update using Kalman update
    # Here is the update in terms that match
    # https://en.wikipedia.org/wiki/Kalman_filter
    # where y has been replaced by w
    H = b.T
    z = ascolumn(mr)
    R = assquare(1/tr)
    P = m2
    x = m1
    w = z - H.dot(x)
    S = R + H.dot(P).dot(H.T)
    #K = P.dot(H.T).dot(inv(S))
    K = linv(S.T,H.dot(P)).T
    x = x + K.dot(w)
    J = I - K.dot(H)
    P = J.dot(P).dot(J.T) + K.dot(R).dot(K.T)
    m1p = ascolumn(x)
    m2p = P
    '''
    '''
    # Here is the update defined in terms of local variables
    # (not renamed to match wikipedia
    vr = 1/tr
    w = ascolumn(mr) - b.T.dot(m1)
    S = assquare(vr) + b.T.dot(m2).dot(b)
    K = linv(S.T,b.T.dot(m2)).T
    m1p = m1 + K.dot(w)
    J = I - K.dot(b.T)
    m2p = J.dot(m2).dot(J.T) + K.dot(assquare(vr)).dot(K.T)
    '''
    '''
    # Here is the update keeping scalars as scalars
    vr  = 1/tr
    w   = mr - scalar(b.T.dot(m1))        # scalar
    S   = vr + scalar(b.T.dot(m2).dot(b)) # scalar
    K   = ascolumn(b.T.dot(m2))/s         # column vector
    m1p = m1 + K*w                        # column vector
    J   = I - K.dot(b.T)                  # matrix
    m2p = J.dot(m2).dot(J.T) + (K.dot(K.T)) * vr
    '''
    '''
    # Here is the update keeping scalars as scalars
    vr = scalar(1/tr)
    w = mr - m       # scalar
    S = vr + v       # scalar
    K = m2b/S  # column vector
    m1p = m1 + K*w
    J = I - K.dot(b.T)
    m2p = J.dot(m2).dot(J.T) + (K*vr).dot(K.T)
    '''
    
    Parameters
    ----------
    m1 : first moment (mean of multivariate gaussian)
    m2 : covariance of multivariate gaussian
    y  : spiking observatoin (scalar)
    b  : projection from multivariate gaussian onto univariate log-rate distribution
    s  : stimulus or bias term for log-rate
    dt : time-step scaling of rate (can also be used as generic gain parameter)
    pm : regularizing prior mean log-rate
    pt : regularizing prior precision
    univariate_method : function, one of
        univariate_lgp_update_moment
        univariate_lgp_update_variational
        univariate_lgp_update_laplace
    """
    # Validate arguments
    if safe:
        m2 = assertfinitereal(assertsquare(m2))
        m1 = assertfinitereal(assertcolumn(m1))
        b  = assertfinitereal(assertcolumn(b))
    # Precompute constants
    K = len(m1)
    I = np.eye(K)
    m2 = 0.5*(m2+m2.T)
    # Gaussian state prior on log-rate
    m2b = m2.dot(b)
    if safe:
        v   = scalar(b.T.dot(m2b))
        m   = scalar(b.T.dot(m1))
    v   = max(eps,(b.T.dot(m2b))[0,0])
    m   = (b.T.dot(m1))[0,0]
    t   = 1/v
    # Regularizing Gaussian prior on log-rate
    tq  = pt + t
    mq = (m*t+pm*pt)/tq
    vq  = 1/tq
    # Integrate to get posterior moments
    mp,vp = univariate_method(mq,vq,y,s,dt)
    if safe:
        mp = scalar(assertfinitereal(mp))
        vp = scalar(assertfinitereal(vp))
    if vp<eps: vp=eps
    tp = 1/vp
    # Generate surrogate univariate likelihood
    tr = max(eps,tp-t)
    vr = scalar(1/tr)
    mr = (mp*tp-m*t)/tr
    if safe:
        mr = scalar(assertfinitereal(mr))
        tr = scalar(assertfinitereal(tr))
    # Futher optimized
    K   = m2b/(vr+v)
    m2p = m2 - K.dot(m2b.T)
    m1p = m1 + K*(mr-m)
    if safe:
        assertfinitereal(m1p)
        assertfinitereal(m2p)
    # Also compute log-likelihood from univariate
    logr   = mp+s
    logPyx = y*logr-sexp(logr)
    ll     = logPyx + 0.5*slog(vp/v) - 0.5*(mp-m)**2/v 

    return m1p, m2p, scalar(ll), mr, vr
    
    
def measurement_update_projected_gaussian_surrogate(m1,m2,y,b,s,dt,pm,pt,
                                      univariate_method = univariate_lgp_update_moment,
                                      return_surrogate=False,
                                      surrogate=None,
                                      eps=1e-12,
                                      safe=False):
    '''
    Parameters
    ----------
    m1 : first moment (mean of multivariate gaussian)
    m2 : covariance of multivariate gaussian
    y  : spiking observatoin (scalar)
    b  : projection from multivariate gaussian onto univariate log-rate distribution
    s  : stimulus or bias term for log-rate
    dt : time-step scaling of rate (can also be used as generic gain parameter)
    pm : regularizing prior mean log-rate
    pt : regularizing prior precision
    univariate_method : function, one of
        univariate_lgp_update_moment
        univariate_lgp_update_variational
        univariate_lgp_update_laplace
    '''
    
    if surrogate is None:
        return measurement_update_projected_gaussian(m1,m2,y,b,s,dt,pm,pt,
                                          univariate_method = univariate_method,
                                          eps=eps,
                                          safe=safe)
    
    (mr, vr) = surrogate
    
    # Validate arguments
    if safe:
        m2 = assertfinitereal(assertsquare(m2))
        m1 = assertfinitereal(assertcolumn(m1))
        b  = assertfinitereal(assertcolumn(b))

    # Precompute constants
    m2 = 0.5*(m2+m2.T)
    # Gaussian state prior on log-rate
    m2b = m2.dot(b)
    if safe:
        v   = scalar(b.T.dot(m2b))
        m   = scalar(b.T.dot(m1))
    v   = max(eps,(b.T.dot(m2b))[0,0])
    m   = (b.T.dot(m1))[0,0]
    t   = 1/v
    # Regularizing Gaussian prior on log-rate
    tq  = pt + t
    mq = (m*t+pm*pt)/tq
    vq  = 1/tq
    #
    tr = 1/vr
    tp = tq + tr
    mp = (mr*tr + mq*tq)/tp
    vp = 1/tp
    if safe:
        mp = scalar(assertfinitereal(mp))
        vp = scalar(assertfinitereal(vp))
    # Futher optimized
    K   = m2b/(vr+v)
    m2p = m2 - K.dot(m2b.T)
    m1p = m1 + K*(mr-m)
    # Also compute log-likelihood from univariate
    logr   = mp+s
    logPyx = y*logr-sexp(logr)
    ll     = logPyx + 0.5*slog(vp/v) - 0.5*(mp-m)**2/v 

    return m1p, m2p, scalar(ll)
