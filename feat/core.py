import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pywt 
from obspy.signal.tf_misfit import cwt as obspycwt



def featurize_cwt(data, dt, nf=1, f_min=1, f_max=50, wl='morlet',w0=5):
    """
    Applies CWT to data and then adds the result to the original data
    
    Parameters
    ------
    data: 1D or 2D Array
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
        Wavelet to use, ex: 'morlet', 'ricker'
    w0: int, default = 5
        parameter for the wavelet, tradeoff between time and frequency resolution
    
    Returns
    -------
    Extended data with CWT, with shape = (len(data, ch_number + ch_number*nf))  
    """ 

    cwt_data = cwt(data=data, dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0, wl=wl)
    extended_data = add_ch(data=data, m_cwt=cwt_data)
        
    return extended_data
    
    

def cwt(data, dt, nf=1, f_min=1, f_max=50, wl='morlet',w0=5):
    """
    Continous Wavelet Transform with 1d and 2d data 
    
    Parameters
    ------
    data: 2D Array  with shape (ch_number, len(data))
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
    Continous wavelet transform applied data shape= (len(data),ch_number*nf)  
    
    
    ch_number: number of channels that data contains
    """
    data = shaper(data)
    params = parameter_calc(wl=wl,dt=dt,f_min=f_min,f_max=f_max,nf=nf,w0=w0)
    ch_number = data.shape[1] 
    length = data.shape[0]
    cwt_ = wavelet_cwt(wl)
    
   
    cwts = np.vstack([cwt_(data[:,ch],**params) for ch in range(ch_number)])  
   
    return cwts.T


def add_ch(data,m_cwt):
    """"
    Adds cwt data to data with multi channel
    Parameters
    ----------
    data : 2D array
           data used in transform
    cwt_data : 2D array
               continous wavelet transform of the data
    Returns
    -------
    2D array data = (len(data), ch*(1+nf))
    """
    data = shaper(data)
    ext_data = np.concatenate((data, m_cwt), axis=1)
    return ext_data





def shaper(data):
    """
    arranges the shape of the input data into the form of (len(data), ch_number)
    if data is 1d then output is (len(data), 1))

    
    Parameters
    ----------
    data : 1D or 2D data 
    
    
    Returns
    -------
    2D array , in the shape of (len(data), ch_number)
    
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = np.reshape(data, (len(data), 1))
    elif data.ndim == 2:
        if data.shape[0] < data.shape[1]:
            data = data.T
    else:
        raise Exception("This function only works with 1d and 2d data")
    return data




def wavelet_cwt(wl):
    """
    preparing wavelet function for transformation
    
    Parameters
    ----------

    wl: str, default = 'morlet'
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


    
def dt_to_widths(dt, f_min, f_max, nf, w0):
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
        Wavelet to use, ex: 'morlet', 'ricker'
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



def parameter_calc(wl, dt, f_min, f_max, nf, w0):
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
        Wavelet to use, ex: 'morlet', 'ricker'
    w0: int, default = 5
        parameter for the wavelet, tradeoff between time and frequency resolution
    
    Returns
    -------
    Parameters as list : params
    """
    if wl == 'ricker':
        widths = dt_to_widths(dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0)
        params = {'wavelet':sp.signal.ricker,'widths':widths}
       
    elif wl =='morlet':
        params = {'dt':dt, 'w0':w0, 'fmin':f_min, 'fmax':f_max, 'nf':nf}        
    else:
        widths = dt_to_widths(dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0)
        params = {'scales':widths, 'wavelet':wl}
    return params
