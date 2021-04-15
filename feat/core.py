from obspy.signal.tf_misfit import cwt as obspycwt
import pywt 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def wavelet(wl):
    """
    preparing wavelet function for transformation
    
    Parameters
    ----------

    wl: str, default = morlet
        to see options, call print_wavelets() functions 

    
    Returns
    -------
    continous wavelet transorm function as cwt_    
    """

    if wl =='morlet':
        cwt_ = obspycwt
    
    elif wl=='ricker':
        cwt_ = sp.signal.cwt
        
    else:
        cwt_ = pywt.cwt
    return cwt_

def cwt(data,
        dt,
        nf=1,
        f_min=1,
        f_max=50,
        wl ='morlet' ,
        w0=5):
    """
    Continous Wavelet Transform
    
    Parameters
    ------
    data: 1D Array
        data to transform
    
    dt: float
        Sample distance in seconds
    nf: int
        Number of logarithmically spaced frequencies between fmin and fmax
    f_min: int, default=1
        Minimum frequency for transformation
    f_max: int , default=50
        Maximum frequency for transformation
    wl: str, default = morlet
        Wavelet to use, ex: morlet, ricker,
    w0: int, default = 5
        parameter for the wavelet, tradeoff between time and frequency resolution
    
    Returns
    -------
    Continous wavelet transform applied data shape= (nf, len(data))
    """
    cwt_ = wavelet(wl)
    params = parameter_calc(wl,dt,f_min,f_max,nf,w0)
    return cwt_(data,*params)
    
def dt_to_widths(dt,f_min,f_max,nf,w0):
    """
    Widths calculator
    
    Parameters
    ----------
    dt: float
        Sample distance in seconds
    nf: int
        Number of logarithmically spaced frequencies between fmin and fmax
    f_min: int, default=1
        Minimum frequency for transformation
    f_max: int , default=50
        Maximum frequency for transformation
    wl: str, default = morlet
        Wavelet to use, ex: morlet, ricker,
    w0: int, default = 5
        parameter for the wavelet, tradeoff between time and frequency resolution
    
    Returns
    -------
    Widths to use
    """
    fs = 1/dt
    freq = np.logspace(np.log10(f_min), np.log10(f_max), nf)
    widths = w0*fs / (2*freq*np.pi)
    return widths



def parameter_calc(wl,dt,f_min,f_max,nf,w0):
    """
    Parameter Calculator for different wavelet functions
    
    Parameters
    ----------
    dt: float
        Sample distance in seconds
    nf: int
        Number of logarithmically spaced frequencies between fmin and fmax
    f_min: int, default=1
        Minimum frequency for transformation
    f_max: int , default=50
        Maximum frequency for transformation
    wl: str, default = morlet
        Wavelet to use, ex: morlet, ricker,
    w0: int, default = 5
        parameter for the wavelet, tradeoff between time and frequency resolution
    
    Returns
    -------
    Parameters as list : params
    """
    if wl == 'ricker':
        widths = dt_to_widths(dt,f_min,f_max,nf,w0)
        params = [sp.signal.ricker,widths]
       
    elif wl =='morlet':
        params = [dt,w0,f_min,f_max,nf]
        
    else:
        widths = dt_to_widths(dt,f_min,f_max,nf,w0)
        params = [widths,wl]
    return params

def print_wavelets():
    """"
    Prints available wavelets
    
    """
    wavlist = ['morlet', 'ricker'] + pywt.wavelist(kind='continuous')
    print(wavlist)
