import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pywt 
from obspy.signal.tf_misfit import cwt as obspycwt



def wavelet_cwt(wl):
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
        wl ='morlet',
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
    cwt_ = wavelet_cwt(wl)
    params = parameter_calc(wl=wl,dt=dt,f_min=f_min,f_max=f_max,nf=nf,w0=w0)
    return cwt_(data,**params)
    
    
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
        widths = dt_to_widths(dt=dt,f_min=f_min,f_max=f_max,nf=nf,w0=w0)
        params = {'wavelet':sp.signal.ricker,'widths':widths}
       
    elif wl =='morlet':
        params = {'dt':dt,'w0':w0,'fmin':f_min,'fmax':f_max,'nf':nf}        
    else:
        widths = dt_to_widths(dt=dt,f_min=f_min,f_max=f_max,nf=nf,w0=w0)
        params = {'scales':widths,'wavelet':wl}
    return params


def print_wavelets():
    """"
    Prints available wavelets
    
    """
    wavlist = ['morlet', 'ricker'] + pywt.wavelist(kind='continuous')
    print(wavlist)
    
def add_wl_ch(data,dt,f_min,f_max,nf,w0,wl):
    """
    Adds CWT applied data to raw data as extra channels. 
    
    Parameters
    ------
    data: 1D or 2D Array
        data to transform and add
    
    dt: float
        Sample distance in seconds
    f_min: int, default=1
        Minimum frequency for transformation
    f_max: int , default=50
        Maximum frequency for transformation
    nf: int
        Number of logarithmically spaced frequencies between fmin and fmax
    w0: int, default = 5
        parameter for the wavelet, tradeoff between time and frequency resolution
    wl: str, default = morlet
        Wavelet to use, ex: morlet, ricker,
    
    
    Returns
    -------
    Raw data extended with Continous wavelet transform applied data = (ch + ch*nf, len(data))
    """
    
    dim = data.ndim

    if dim == 1:
        
        ch = 1
        length = data.shape[0]
        
        cwt_op = cwt(data,dt=dt,f_min=f_min,f_max=f_max,nf=nf,w0=w0,wl=wl)
        
        ext_data = np.zeros((ch+nf, length))
        ext_data[:ch, :] = data
        ext_data[ch:, :] = cwt_op
    
    elif dim == 2:
        
        ch = data.shape[0]
        length = data.shape[1]
        
        cwt_list = []
        
        for curr_ch in range(ch):
            ch_data = data[curr_ch]
            cwt_op = cwt(ch_data,dt=dt,f_min=f_min,f_max=f_max,nf=nf,w0=w0,wl=wl)
            cwt_list.append(cwt_op)
        
        cwt_list = np.asarray(cwt_list)
        cwt_list = np.resize(cwt_list, (ch*nf, length))
        ext_data = np.concatenate((data, cwt_list), axis=0)
             
    return ext_data



def plt_ch(ext_data):
    
    time = np.arange(0, ext_data.shape[1])

    for i in range(ext_data.shape[0]):
        plt.plot(time, ext_data[i])
        plt.show()
