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
    Extended data with CWT, if ndim = 1 : shape = (1+nf, len(data))
    otherwise shape = (ch_number + ch_number*nf, len(data))  
    """ 
    if data.ndim == 1:
        
        cwt_data = cwt(data=data, dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0, wl=wl)
        extended_data = add_1ch(data=data,cwt_data=cwt_data)
    elif data.ndim == 2:
        cwt_data = multi_cwt(data=data, dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0, wl=wl)
        extended_data = add_multich(data=data, m_cwt=cwt_data)
        
    return extended_data
    
    

def cwt(data, dt, nf=1, f_min=1, f_max=50, wl='morlet', w0=5):
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
        Wavelet to use, ex: 'morlet', 'ricker'
    w0: int, default = 5
        parameter for the wavelet, tradeoff between time and frequency resolution
    
    Returns
    -------
    Continous wavelet transform applied data shape= (nf, len(data))
    """
    cwt_ = wavelet_cwt(wl)
    params = parameter_calc(wl=wl, dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0)
    return cwt_(data, **params)
    


def multi_cwt(data, dt, nf=1, f_min=1, f_max=50, wl='morlet', w0=5):
    """
    Continous Wavelet Transform with multichannel data 
    
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
        Wavelet to use, ex: 'morlet', 'ricker'
    w0: int, default = 5
        parameter for the wavelet, tradeoff between time and frequency resolution
    

    Returns
    -------
    Continous wavelet transform applied data shape= (len(data),ch_number*nf)  
    
    
    ch_number: number of channels that data contains
    """
    data = shaper(data)
    ch_number= data.shape[1]
    length = data.shape[0]
    
    cwt_list = []
    for ch in range(ch_number):
        ch_data = data[:, ch]
        cwt_op = cwt(ch_data, dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0, wl=wl)
        cwt_list.append(cwt_op)
        
    cwt_ = np.asarray(cwt_list)
    cwt_ = np.reshape(cwt_, (length, ch_number*nf))
    return cwt_




def add_multich(data, m_cwt):
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
    2D array data = (1+nf, len(data))
    """
    ext_data = np.concatenate((data, m_cwt), axis=0)
    return ext_data



def add_1ch(data, cwt_data):
    """"
    Adds cwt data to data with 1 channel
    
    Parameters
    ----------
    data : 1D array
           data used in transform
    
    cwt_data : 2D array
               continous wavelet transform of the data
    
    Returns
    -------
    2D array data = (1+nf, len(data))
    
    """
    
    if data.ndim == 1:
        length = data.shape[0]
    else:
        length = data.shape[1]
    ch = 1
    nf = cwt_data.shape[0]
    
    ext_data = np.zeros((ch+nf, length))
    ext_data[:ch, :] = data
    ext_data[ch:, :] = cwt_data
    return ext_data

def shaper(data):

    data = np.asarray(data)
    if data.ndim == 1:
        data = np.reshape(data, (len(data), 1))
    elif data.ndim == 2:
        if data.shape[0] < data.shape[1]:
            data = np.reshape(data, (data.shape[1], data.shape[0]))
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


    








def plt_ch(ext_data):
    
    time = np.arange(0, ext_data.shape[1])

    for i in range(ext_data.shape[0]):
        plt.plot(time, ext_data[i])
        plt.show()
        
 
def print_wavelets():
    """"
    Prints available wavelets
    
    """
    wavlist = ['morlet', 'ricker'] + pywt.wavelist(kind='continuous')
    print(wavlist)
