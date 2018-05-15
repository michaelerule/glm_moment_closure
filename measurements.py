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
import traceback
import functions

from scipy.linalg import lstsq,pinv
from numpy.linalg.linalg import cholesky as chol
from numpy.linalg.linalg import LinAlgError

from functions import sexp,slog
from utilities import *
from arguments import *
from utilities import current_milli_time
from plot      import v2str_long

def minimize_retry(objective,initial,jac=None,hess=None,
                   verbose=False,
                   printerrors=True,
                   failthrough=True,
                   tol=1e-5,
                   simplex_only=False,
                   show_progress=True,
                   **kwargs):
    '''
    Call `scipy.optimize.minimize`, retrying a few times in case
    one solver doesn't work.
    
    This addresses unresolved bugs that can cause exceptions in some of
    the gradient-based solvers in Scipy. If we happen upon these bugs, 
    we can continue optimization using slower but more robused methods. 
    
    Ultimately, this routine falls-back to the gradient-free Nelder-Mead
    simplex algorithm, although it will try to use faster routines if
    the hessian and gradient are providede. 
    '''
    # Store and track result so we can keep best value, even if it crashes
    result = None
    x0     = np.array(initial).ravel()
    g0     = 1/np.zeros(x0.shape)
    nfeval = 0
    ngeval = 0
    if jac is True:
        v,g  = objective(x0)
    else:
        v    = objective(x0)
    best = v
    # Show progress of the optimization?
    if show_progress:
        sys.stdout.write('\n')
        last_shown = current_milli_time()
    def progress_update():
        nonlocal best, x0, nfeval, ngeval, last_shown
        if show_progress: 
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
    # Wrap the provided gradient and objective functions, so that we can
    # capture the function values as they are being optimized. This way, 
    # if optimization throws an exception, we can still remember the best
    # value it obtained, and resume optimization from there using a slower
    # but more reliable method. These wrapper functions also act as 
    # callbacks and allow us to print the optimization progress on screen.
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
    # There are still some unresolved bugs in some of the optimizers that
    # can lead to exceptions and crashes! This routine catches these errors
    # and failes gracefully. Note that system interrupts are not caught, 
    # and other unexpected errors are caught but reported, in case they
    # reflect an exception arising from a user-provided gradient or 
    # objective function.
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
            # Don't catch system interrupts
            raise
        except (TypeError,NameError):
            # Likely an internal bug in scipy; don't report it
            clear_progress()
            return False
        except Exception:
            # Unexpected error, might be user error, report it
            traceback.print_exc()
            clear_progress()
            if verbose or printerrors:
                sys.stderr.write('Error using minimize with %s:\n'%method)
                sys.stderr.flush()
                traceback.print_exc()
                sys.stderr.flush()
            return False
        return False
    # We try a few different optimization, in order
    # -- If Hessian is available, Newton-CG should be fast! try it
    # -- Otherwise, BFGS is a fast gradient-only optimizer
    # -- Fall back to Nelder-Mead simplex algorithm if all else fails
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
        print('Best parameters are %s with value %s'%(v2str_long(x0),best))
        raise
    except Exception:
        traceback.print_exc()
        if not failthrough: raise
    # If we've reached here, it means that all optimizers terminated with
    # an error, or reported a failure to converge. If `failthrough` is 
    # set, we can still return the best value found so far. 
    if failthrough:
        if verbose:
            sys.stderr.write('Minimization may not have converged\n')
            sys.stderr.flush()
        return x0 # fail through
    raise ArithmeticError('All minimization attempts failed')

def univariate_lgp_update_moment(m,v,y,s,dt,
                    tol      = 1e-3,
                    maxiter  = 100,
                    eps      = 1e-7,
                    minlrate = -200,
                    maxlrate = 20,
                    ngrid    = 50,
                    minprec  = 1e-6,
                    maxrange = 150):
    '''
    Update a log-Gaussian distribution with a Poisson measurement by
    integrating to extract the posterior mean and variance.
    '''
    # Get moments by integrating
    v = max(v,eps)
    t = 1/v
    m = np.clip(m,minlrate,maxlrate)
    # Set integration limits
    m0,s0 = (m,np.sqrt(v)) if t>minprec else (slog(y+0.25),np.sqrt(1/(y+1)))
    m0 = np.clip(m0,minlrate,maxlrate)
    delta = min(4*s0,maxrange)
    x = np.linspace(m0-delta,m0+delta,ngrid)
    # Calculate likelihood contribution
    r  = x + s + slog(dt)
    ll = y*r-sexp(r)
    # Calculate prior contribution
    lq = -.5*((x-m)**2/v+slog(2*np.pi*v))
    # "clean up" prior (numerical stability)
    q  = np.maximum(eps,sexp(lq))
    q  = q/np.sum(q)
    lq = slog(q)
    # Normalize to prevent overflow, calculate log-posterior
    nn = np.max(ll)
    lp = (ll - nn) + lq
    # Estimate posterior
    p  = np.maximum(eps,sexp(lp))
    s  = np.sum(p)
    p /= s
    # Integrate to get posterior moments and likelihood
    pm = np.sum(x*p)
    pv = np.sum((x-pm)**2*p)
    ll = scipy.misc.logsumexp(ll+lq)
    assertfinitereal(pm)
    assertfinitereal(pv)
    assertfinitereal(ll)
    return pm,pv,ll

def univariate_lgp_update_laplace(m,v,y,s,dt,
                    tol     = 1e-6,
                    maxiter = 20,
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
    mu = minimize_retry(objective,m,gradient,hessian,tol=tol,show_progress=False,printerrors=False)
    vv = 1/hessian(mu)
    # Get likelihood at posterior mode
    logr   = mu+s+slog(dt)
    logPyx = y*logr-sexp(logr)
    ll     = logPyx + 0.5*slog(vv/v) - 0.5*(mu-m)**2/v 
    return mu, vv, ll

def univariate_lgp_update_variational(m,v,y,s,dt,
                    tol     = 1e-6,
                    maxiter = 20,
                    eps     = 1e-12):
    '''
    Optimize variational approximation
    Mean and variance must be optimized jointly
    Log-Gaussian / Poisson model
    '''
    v = max(v,eps)
    scale = dt*sexp(s)
    # Use minimization to solve for variational solution
    def objective(parameters):
        mq,vq = parameters
        rate  = scale*np.exp(mq+ 0.5*vq)
        if vq<eps or not np.isfinite(mq) or not np.isfinite(vq) or not np.isfinite(rate):
            return np.inf
        return -y*mq + rate + 0.5*( -slog(vq) + vq/v + (mq-m)**2/v )
    def gradient(parameters):
        mq,vq = parameters
        rate  = scale*np.exp(mq+ 0.5*vq)
        if vq<eps or not np.isfinite(mq) or not np.isfinite(vq) or not np.isfinite(rate):
            return (np.NaN,np.NaN)
        dm    = -y + rate + (mq-m)/v
        dv    = rate*0.5 + 0.5*(1/v-1/vq)
        return np.array([dm, dv]).squeeze()
    def hessian(parameters):
        mq,vq = parameters
        rate  = scale*np.exp(mq+ 0.5*vq)
        if vq<eps or not np.isfinite(mq) or not np.isfinite(vq) or not np.isfinite(rate): 
            return [[np.NaN,np.NaN],[np.NaN,np.NaN]]
        dmdm  = rate + 1/v
        dvdm  = rate*0.5
        dvdv  = rate*0.25 + 0.5/vq**2
        return np.array([[dmdm,dvdm],[dvdm,dvdv]]).squeeze()
    mp,vp = minimize_retry(objective,[m,v],gradient,hessian,tol=tol,show_progress=False,printerrors=False)
    # Get likelihood using log-Gaussian assumption
    logr   = mp+s+slog(dt)
    logPyx = y*logr-sexp(logr+0.5*vp)
    ll     = logPyx + 0.5*slog(vp/v) - 0.5*(mp-m)**2/v 
    return mp, vp, ll
    
def univariate_lgp_update_variational_so(m,v,y,s,dt,
                    tol     = 1e-6,
                    maxiter = 20,
                    eps     = 1e-12):
    '''
    Optimize variational approximation
    Mean and variance must be optimized jointly
    2nd order Gaussian/Poisson model approximation
    '''
    v = max(v,eps)
    scale = dt*sexp(s)
    # Use minimization to solve for variational solution
    def objective(parameters):
        mq,vq = parameters
        rate  = scale*np.exp(mq)*(1 + 0.5*vq)
        if vq<eps or not np.isfinite(mq) or not np.isfinite(vq) or not np.isfinite(rate):
            return np.inf
        return -y*mq + rate + 0.5*( -slog(vq) + vq/v + (mq-m)**2/v )
    def gradient(parameters):
        mq,vq = parameters
        rate  = scale*np.exp(mq)*(1 + 0.5*vq)
        if vq<eps or not np.isfinite(mq) or not np.isfinite(vq) or not np.isfinite(rate):
            return (np.NaN,np.NaN)
        dm    = -y + rate + (mq-m)/v
        dv    = rate*0.5 + 0.5*(1/v-1/vq)
        return np.array([dm, dv]).squeeze()
    def hessian(parameters):
        mq,vq = parameters
        rate  = scale*np.exp(mq)*(1 + 0.5*vq)
        if vq<eps or not np.isfinite(mq) or not np.isfinite(vq) or not np.isfinite(rate): 
            return [[np.NaN,np.NaN],[np.NaN,np.NaN]]
        dmdm  = rate + 1/v
        dvdm  = rate*0.5
        dvdv  = rate*0.25 + 0.5/vq**2
        return np.array([[dmdm,dvdm],[dvdm,dvdv]]).squeeze()
    mp,vp = minimize_retry(objective,[m,v],gradient,hessian,tol=tol,show_progress=False,printerrors=False)
    # Get likelihood using second order assumption
    logr   = mp+s+slog(dt)
    logPyx = y*logr-sexp(logr)*(1+0.5*vp)
    ll     = logPyx + 0.5*slog(vp/v) - 0.5*(mp-m)**2/v 
    return mp, vp, ll

def measurement_update_projected_gaussian(m1,m2,y,b,s,dt,pm,pt,
                                          univariate_method = univariate_lgp_update_moment,
                                          eps=1e-7,
                                          safe=False):
    """
    This performs an approximation to the non-conjudate log-Gaussian
    Poisson measurement update, and then propagates the result of this 
    update to the full jointly-Gaussian model of the history process.
    
    The non-conjugate update can be performed via one of several 
    approximation functions, including
    
     - moment matching (`univariate_lgp_update_moment`)
     - Laplace approximation (`univariate_lgp_update_laplace`)
     - Variational with log-Gaussian assumption (`univariate_lgp_update_variational`)
     - Variational with second-order moment assumption (`univariate_lgp_update_variational_so`)
    
    Propagation to the full jointly Gaussian history model can be thought
    of as multiplying the updated marginal Gaussian by a conditional
    gaussian representing the subspace of the prior that is orthogonal to
    the updated projection. 
    
    This can also be thought of as constructing a "surrogate" gaussian
    likelihood representing the shifted mean and added precision introduced
    by the measurement, and then performing a Kalman-filter-style update
    on the full jointly Gaussian latent space. 
    
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
    mp,vp,ll = univariate_method(mq,vq,y,s,dt)
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
    # Optimized Kalman update (speed over stability)
    K   = m2b/(vr+v)
    m2p = m2 - K.dot(m2b.T)
    m1p = m1 + K*(mr-m)
    if safe:
        assertfinitereal(m1p)
        assertfinitereal(m2p)
    # Compute log-likelihood from univariate
    #logr   = mp+s
    #logPyx = y*logr-sexp(logr)
    #ll     = logPyx + 0.5*slog(vp/v) - 0.5*(mp-m)**2/v 
    return m1p, m2p, scalar(ll), mr, vr
    
    
def measurement_update_projected_gaussian_surrogate(m1,m2,y,b,s,dt,pm,pt,
                                      univariate_method = univariate_lgp_update_moment,
                                      return_surrogate=False,
                                      surrogate=None,
                                      eps=1e-12,
                                      safe=False):
    '''
    Please see `measurement_update_projected_gaussian`. 
    
    This function is the same, except that it can return "surrogate"
    gaussian approximations of the measurement updates, which can then
    be used in subsequent filtering of the same data for much more rapid
    measurement updates.
    
    If the parameters do not change too much, these surrogate updates
    will remain approximately correct. This provides a path toward an 
    approximate EM-style algorithm for optimizing the likelihood using
    moment-closures as an additional likelihood penalty (regularizer) for
    slow dynamics.
    
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
