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
if sys.version_info<(3,):
    from itertools import imap as map

import os
import pickle
import scipy
import numpy
from   matplotlib.pyplot import *
import matplotlib.pyplot as plt
from   matplotlib.pylab  import find

try: # python 2.x
    from itertools import izip, chain
except: # python 3
    from itertools import chain
    izip = zip

try:
    import statsmodels
    import statsmodels.api as smapi
    import statsmodels.graphics as smgraphics
except:
    print('could not find statsmodels; some plotting functions missing')


# This is the color scheme from the painting "gather" by bridget riley
GATHER = [
'#f1f0e9', # "White"
'#eb7a59', # "Rust"
'#eea300', # "Sand"
'#5aa0df', # "Azure"
'#00bac9', # "Turquoise"
'#44525c'] # "Black"
WHITE,RUST,OCHRE,AZURE,TURQUOISE,BLACK = GATHER

# Other colors
GREEN  = '#77ae64'
MAUVE  = '#956f9b'
INDEGO     = [.37843053,  .4296282 ,  .76422011]
VERIDIAN   = [.06695279,  .74361409,  .55425139]
CHARTREUSE = [.71152929,  .62526339,  .10289384]
CRIMSON    = [.84309675,  .37806273,  .32147779]


def simpleaxis(ax=None):
    '''
    Only draw the bottom and left axis lines
    
    Parameters
    ----------
    ax : optiona, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def simpleraxis(ax=None):
    '''
    Only draw the left y axis, nothing else
    
    Parameters
    ----------
    ax : optiona, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def noaxis(ax=None):
    '''
    Hide all axes
    
    Parameters
    ----------
    ax : optiona, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def nicey():
    '''
    Mark only the min/max value of y axis
    '''
    if ylim()[0]<0:
        yticks([ylim()[0],0,ylim()[1]])
    else:
        yticks([ylim()[0],ylim()[1]])

def nicex():
    '''
    Mark only the min/max value of x axis
    '''
    if xlim()[0]<0:
        xticks([xlim()[0],0,xlim()[1]])
    else:
        xticks([xlim()[0],xlim()[1]])

def nicexy():
    '''
    Mark only the min/max value of y/y axis. See `nicex` and `nicey`
    '''
    nicey()
    nicex()

def positivex():
    '''
    Sets the lower x limit to zero, and the upper limit to the largest
    positive value un the current xlimit. If the curent xlim() is
    negative, a value error is raised.
    '''
    top = np.max(xlim())
    if top<=0:
        raise ValueError('Current axis view lies within negative '+
            'numbers, cannot crop to a positive range')
    xlim(0,top)
    nicex()

def positivey():
    '''
    Sets the lower y limit to zero, and the upper limit to the largest
    positive value un the current ylimit. If the curent ylim() is
    negative, a value error is raised.
    '''
    top = np.max(ylim())
    if top<=0:
        raise ValueError('Current axis view lies within negative '+
            'numbers, cannot crop to a positive range')
    ylim(0,top)
    nicey()

def positivexy():
    '''
    Remove negative range from both x and y axes. See `positivex` and
    `positivey`
    '''
    positivex()
    positivey()

def xylim(a,b,ax=None):
    '''
    set x and y axis limits to the smae range
    
    Parameters
    ----------
    a : lower limit
    b : upper limit
    '''
    if ax==None: ax = plt.gca()
    ax.set_xlim(a,b)
    ax.set_ylim(a,b)

def nox():
    '''
    Hide x-axis
    '''
    xticks([])
    xlabel('')

def noy():
    '''
    Hide y-axis
    '''
    yticks([])
    ylabel('')

def noxyaxes():
    '''
    Hide all aspects of x and y axes. See `nox`, `noy`, and `noaxis`
    '''
    nox()
    noy()
    noaxis()

def righty(ax=None):
    '''
    Move the y-axis to the right
    '''
    if ax==None: ax=plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

def unity():
    '''
    Set y-axis to unit interval
    '''
    ylim(0,1)
    nicey()

def unitx():
    '''
    Set x-axis to unit interval
    '''
    xlim(0,1)
    nicex()

def force_aspect(aspect=1,a=None):
    '''
    Parameters
    ----------
    aspect : aspect ratio
    '''
    if a is None: a = plt.gca()
    x1,x2=a.get_xlim()
    y1,y2=a.get_ylim()
    a.set_aspect(np.abs((x2-x1)/(y2-y1))/aspect)

def get_ax_size(ax=None,fig=None):
    '''
    Gets tha axis size in figure-relative units
    '''
    if fig is None: fig = plt.gcf()
    if ax is None: ax  = plt.gca()
    '''http://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels'''
    fig  = plt.gcf()
    ax   = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width  *= fig.dpi
    height *= fig.dpi
    return width, height

def get_ax_pixel(ax=None,fig=None):
    '''
    Gets tha axis size in pixels
    '''
    if fig is None: fig = plt.gcf()
    if ax is None: ax  = plt.gca()
    # w/h in pixels
    w,h = get_ax_size()
    # one px in axis units is the axis span div no. pix
    dy = np.diff(ylim())
    dx = np.diff(xlim())
    return dx/float(w),dy/float(h)

def pixels_to_xunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current x-axis
    scale
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dx  = np.diff(plt.xlim())[0]
    return n*dx/float(w)

def yunits_to_pixels(n,ax=None,fig=None):
    '''
    Converts a measurement in units of the current y-axis to pixels
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dy  = np.diff(plt.ylim())[0]
    return n*float(h)/dy

def xunits_to_pixels(n,ax=None,fig=None):
    '''
    Converts a measurement in units of the current x-axis to pixels
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dx = np.diff(xlim())[0]
    return n*float(w)/dx

def pixels_to_yunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current y-axis
    scale
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dy = np.diff(ylim())[0]
    return n*dy/float(h)

def pixels_to_xfigureunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current
    figure width scale
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w_pixels = fig.get_size_inches()[0]*fig.dpi
    return n/float(w_pixels)

def pixels_to_yfigureunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current
    figure height scale
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    h_pixels = fig.get_size_inches()[1]*fig.dpi
    return n/float(h_pixels)

def nudge_axis_y_pixels(dy,ax=None):
    '''
    moves axis dx pixels.
    Direction of dx may depent on axis orientation. TODO: fix this
    '''
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = -pixels_to_yfigureunits(float(dy),ax)
    ax.set_position((x,y-dy,w,h))

def adjust_axis_height_pixels(dy,ax=None):
    '''
    moves axis dx pixels.
    Direction of dx may depent on axis orientation. TODO: fix this
    '''
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    ax.set_position((x,y,w,h-pixels_to_yfigureunits(float(dy),ax)))

def nudge_axis_y(dy,ax=None):
    '''
    Parameters
    ----------
    dy : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,ax)
    ax.set_position((x,y+dy,w,h))

def nudge_axis_x(dx,ax=None):
    '''
    Parameters
    ----------
    dx : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,ax)
    ax.set_position((x+dx,y,w,h))

def expand_axis_y(dy,ax=None):
    '''
    Parameters
    ----------
    dy : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,ax)
    ax.set_position((x,y,w,h+dy))

def nudge_axis_baseline(dy,ax=None):
    '''
    Parameters
    ----------
    dy : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = pixels_to_yfigureunits(dy,ax)
    ax.set_position((x,y+dy,w,h-dy))

def nudge_axis_left(dx,ax=None):
    '''
    Parameters
    ----------
    dx : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,ax)
    ax.set_position((x+dx,y,w-dx,h))

def fudgex(by=10,ax=None,doshow=False):
    '''
    Move x axis label closer    
    '''
    if ax is None: ax=plt.gca()
    ax.xaxis.labelpad = -by
    plt.draw()
    if doshow:
        plt.show()

def fudgey(by=20,ax=None,doshow=False):
    '''
    Move y axis label closer
    '''
    if ax is None: ax=plt.gca()
    ax.yaxis.labelpad = -by
    plt.draw()
    if doshow:
        plt.show()

def fudgexy(by=10,ax=None):
    '''
    Move x and y axis label closer
    '''
    fudgex(by,ax)
    fudgey(by,ax)

def subfigurelabel(x,subplot_label_size=14,dx=20,dy=5):
    '''
    Parameters
    ----------
    x : label
    '''
    fontproperties = {
        'family':'Bitstream Vera Sans',
        'weight': 'bold',
        'size': subplot_label_size,
        'verticalalignment':'bottom',
        'horizontalalignment':'right'}
    text(xlim()[0]-pixels_to_xunits(dx),ylim()[1]+pixels_to_yunits(dy),x,**fontproperties)

def shortscientific(x,prec=0):
    '''
    Parameters
    ----------
    x : scalar numeric
    prec : non-negative integer
    
    Returns
    -------
    '''
    return ('%.*e'%(prec,x)).replace('-0','-').replace('+','').replace('e0','e')

def sigbar(x1,x2,y,pvalue,dy=5,LABELSIZE=10):
    '''
    draw a significance bar between position x1 and x2 at height y 
    
    Parameters
    ----------
    '''
    dy = pixels_to_yunits(dy)
    height = y+2*dy
    plot([x1,x1,x2,x2],[height-dy,height,height,height-dy],lw=0.5,color=BLACK)
    text(np.mean([x1,x2]),height+dy,shortscientific(pvalue),fontsize=LABELSIZE,horizontalalignment='center')

def v2str(p):
    '''
    Format vector as string in short scientific notation
    '''
    return '['+','.join([shortscientific(x) for x in p])+']'

def v2str_long(p):
    '''
    Format vector as string with maximum precision
    '''
    return '['+','.join([np.float128(x).astype(str) for x in p])+']'

import datetime
def today():
    '''
    Returns
    -------
    `string` : the date in YYMMDD format
    '''
    return datetime.date.today().strftime('%Y%m%d')

def savefigure(name):
    '''
    Saves figure as both SVG and PDF, prepending the current date
    in YYYYMMDD format
    
    Parameters
    ----------
    name : string
        file name to save as (sans extension)
    '''
    # strip user-supplied extension if present
    dirname  = os.path.dirname(name)
    if dirname=='': dirname='./'
    basename = os.path.basename(name)
    if basename.split('.')[-1].lower() in {'svg','pdf','png'}:
        basename = '.'.join(basename.split('.')[:-1])
    savefig(dirname + os.path.sep + today()+'_'+basename+'.svg',transparent=True,bbox_inches='tight')
    savefig(dirname + os.path.sep + today()+'_'+basename+'.pdf',transparent=True,bbox_inches='tight')
    savefig(dirname + os.path.sep + today()+'_'+basename+'.png',transparent=True,bbox_inches='tight')

def clean_y_range(ax=None,precision=1):
    '''
    Round down to a specified number of significant figures
    '''
    if ax is None: ax=plt.gca()
    y1,y2 = ylim()
    precision = 10.0**precision
    _y1 = floor(y1*precision)/precision
    _y2 = ceil (y2*precision)/precision
    ylim(min(_y1,ylim()[0]),max(_y2,ylim()[1]))

def round_to_precision(x,precision=1):
    '''
    Round to a specified number of significant figures
    
    Parameters
    ----------
    x : scalar
        Number to round
    precision : positive integer, default=1
        Number of digits to keep
    
    Returns
    -------
    x : scalar
        Rounded number
    '''
    if x==0.0: return 0
    magnitude = np.abs(x)
    digits    = np.ceil(np.log10(magnitude))
    factor    = 10.0**(precision-digits)
    return np.round(x*factor)/factor

def ceil_to_precision(x,precision=1):
    '''
    Round up to a specified number of significant figures
    
    
    Parameters
    ----------
    x : scalar
        Number to round
    precision : positive integer, default=1
        Number of digits to keep
    
    Returns
    -------
    x : scalar
        Rounded number
    -------
    '''
    if x==0.0: return 0
    magnitude = np.abs(x)
    digits = np.ceil(np.log10(magnitude))
    factor = 10.0**(precision-digits)
    precision *= factor
    return np.ceil(x*precision)/precision

def floor_to_precision(x,precision=1):
    '''
    Round down to a specified number of significant figures
    
    Parameters
    ----------
    x : scalar
        Number to round
    precision : positive integer, default=1
        Number of digits to keep
    
    Returns
    -------
    x : scalar
        Rounded number
    '''
    if x==0.0: return 0
    magnitude = np.abs(x)
    digits = np.ceil(np.log10(magnitude))
    factor = 10.0**(precision-digits)
    precision *= factor
    return np.floor(x*precision)/precision




def expand_y_range(yvalues,ax=None,precision=1,pad=1.2):
    '''
    Round upper/lower y axis bound outwards to given precision
    '''
    if ax is None: ax=plt.gca()
    yy = np.array(yvalues)
    m = np.mean(yy)
    yy = (yy-m)*pad+m
    y1 = np.min(yy)
    y2 = np.max(yy)
    precision = 10.0**precision
    _y1 = floor_to_precision(y1,precision)
    _y2 = ceil_to_precision(y2,precision)
    ylim(min(_y1,ylim()[0]),max(_y2,ylim()[1]))

def stderrplot(m,v,color='k',alpha=0.1,smooth=None,lw=1.5,filled=True,label=None,stdwidth=1.96,**kwargs):
    '''
    Parameters
    ----------
    m : mean
    v : variance
    
    Other Parameters
    ----------------
    color : 
        Plot color
    alpha : 
        Shaded confidence alpha color blending value
    smooth : int
        Number of samples over which to smooth the variance
    '''
    plot(m, color = color,lw=lw,label=label)
    e = np.sqrt(v)*stdwidth
    if not smooth is None and smooth>0:
        e = neurotools.signal.signal.box_filter(e,smooth)
        m = neurotools.signal.signal.box_filter(m,smooth)
    if filled:
        c = matplotlib.colors.colorConverter.to_rgb(color)+(alpha ,)
        fill_between(np.arange(len(m)),m-e,m+e,lw=0,color=c,**kwargs)
    else:
        plot(m-e,':',lw=lw*0.5,color=color,**kwargs)
        plot(m+e,':',lw=lw*0.5,color=color,**kwargs)    

def yscalebar(ycenter,yheight,label,x=None,color='k',fontsize=9,ax=None):
    '''
    Add vertical scale bar to plot
    '''
    yspan = [ycenter-yheight/2.0,ycenter+yheight/2.0]
    if ax is None:
        ax = plt.gca()
    plt.draw() # enforce packing of geometry
    if x is None:
        x = -pixels_to_xunits(5)
    plt.plot([x,x],yspan,color='k',lw=1,clip_on=False)
    plt.text(x-pixels_to_xunits(2),np.mean(yspan),label,
        rotation=90,
        fontsize=9,
        horizontalalignment='right',
        verticalalignment='center')
        
def addspikes(Y,lw=0.2,color='k'):
    '''
    Add vertical lines where Y>0
    '''
    for t in find(Y>0): 
        axvline(t,lw=lw,color=color)
