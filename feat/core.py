import numpy as np
import scipy as sp
import pywt 
from obspy.signal.tf_misfit import cwt as obspycwt
from functools import partial



def featurize_cwt(data, dt, nf=None, f_min=None, f_max=None, wl='morlet', w0=None, widths=None):
    """
    Applies CWT to data and then adds the result to the original data.
    There are two options for CWT:
    1- Frequency limits: define f_min, f_max and nf ---> wavelets: Morlets and Mexh
    2- Widths: define widths, do not define f_min, f_max and nf ---> wavelets: All of them.
    
    Parameters
    ------
    data: 1D or 2D Array
        Data to transform
    
    dt: float
        Sample distance in seconds
    nf: int
        Number of logarithmically spaced frequencies between fmin and fmax
    f_min: int, default = None
        Minimum frequency for transformation
    f_max: int, default = None
        Maximum frequency for transformation
    wl: str, default = 'morlet'
        Wavelet to use, Obspy, Scipy, and Pywt wavelets. Use list_wavelets() to see options.
    w0: int, default = None
        Parameter for the wavelet, tradeoff between time and frequency resolution
    widths: array_like, default = None
        The wavelet scales to use. Wavelets other than morlet
        and mexh needs widths parameter.
      
        Optional choice to use widths instead of frequency limits.
        
    Returns
    -------
    Extended data with CWT, with shape = (len(data, num_channels + num_channels*nf)) if used frequency option.
     with widths option nf is replaced by len(widths). 
    """ 

    cwt_data = cwt(data=data, dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0, wl=wl, widths=widths)
    extended_data = add_ch(data=data, m_cwt=cwt_data)
        
    return extended_data
    
    

def cwt(data, dt, nf=None, f_min=None, f_max=None, wl='morlet', w0=None, widths=None):
    """
    Continous Wavelet Transform using 1D wavelets with 1D or 2D data 
    There are two options for CWT:
    1- Frequency limits: define f_min, f_max and nf ---> wavelets: Morlets and Mexh
    2- Widths: define widths, do not define f_min, f_max and nf ---> wavelets: All of them.
    
    Parameters
    ------
    data: 1D or 2D Array  with shape (num_samples, num_channels)
        Data to transform
    
    dt: float
        Sample distance in seconds
    nf: int
        Number of logarithmically spaced frequencies between fmin and fmax
    f_min: int, default = None
        Minimum frequency for transformation
    f_max: int , default = None
        Maximum frequency for transformation
    wl: str, default = 'morlet'
        Wavelet to use, Obspy, Scipy, and Pywt wavelets. Use list_wavelets() to see options.
    w0: int, default = None
        Parameter for the wavelet, tradeoff between time and frequency resolution
    widths: array_like, default = None
        The wavelet scales to use. Wavelets other than morlet
        and mexh needs widths parameter.
      
        
    
    Returns
    -------
    Continous wavelet transform applied data shape= (num_samples, num_channels*nf)  if used frequency option.
     with widths option nf is replaced by len(widths).
    
    
    num_channels: Number of channels that data contains
    """
    
    data = _shaper(data)
    params = calc_parameter(wl=wl, dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0, widths=widths)
    num_channels = data.shape[1] 
    cwt_ = get_cwt_fn(wl)
    if wl not in ['morlet', 'morlet_s', 'ricker']:
        cwts = np.vstack([cwt_(data = data[:, ch], **params)[0] for ch in range(num_channels)])  
        # Here we seperate pywt wavelet functions from others. Because pywt.cwt function gives two outputs
        # (transformed_data, frequencies). We only need transformed_data, so we took only the first element 
        # of the output.
    else:
        cwts = np.vstack([cwt_(data[:, ch], **params) for ch in range(num_channels)])  
   
    return cwts.T


def add_ch(data, m_cwt):
    """"
    Adds cwt data to data with multi channel
    Parameters
    ----------
    data : 1D or 2D array
           Data used in transform
    cwt_data : 2D array
           Continous wavelet transform of the data
    Returns
    -------
    2D array data = (num_samples, ch*(1+nf)) if used frequency option.
     with widths option nf is replaced by len(widths).
    """
    data = _shaper(data)
    ext_data = np.concatenate((data, m_cwt), axis=1)
    return ext_data





def _shaper(data):
    """
    Checks the shape and dimension of the data. If data is 1d transforms it to 2d 
    
    Parameters
    ----------
    data : 1D or 2D data 
    
    
    Returns
    -------
    2D array, in the shape of (num_samples, num_channels)
    
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = np.reshape(data, (len(data), 1))
    elif data.ndim == 2:
        assert data.shape[0] > data.shape[1]
    else:
        raise ValueError("This function only works with 1D and 2D data")
    return data





def get_cwt_fn(wl):
    """
    preparing wavelet function for transformation
    
    Parameters
    ----------

    wl: str, default = 'morlet'
        To see options, call list_wavelets() function

    
    Returns
    -------
    continous wavelet transorm function as cwt_    
    """

    if wl =='morlet':
        cwt_ = partial(obspycwt,wl='morlet')         # Obspy has only morlet wavelet for cwt
    
    elif wl=='ricker':
        cwt_ = partial(sp.signal.cwt,wavelet=sp.signal.ricker) 
                                # Scipy has ricker and morlet wavelet that works on cwt
        
    elif wl=='morlet_s':       
        cwt_ = partial(sp.signal.cwt,wavelet=sp.signal.morlet2)  
    else:
        cwt_ = partial(pywt.cwt,wavelet=wl)        # All pywt wavelets can be seen with 
        			 # print_wavelets() function from utils.
    return cwt_



    
def _calc_widths(dt, f_min, f_max, nf, wl, w0, widths):
    """
    Widths calculator for morlet and ricker wavelet 
    
    Parameters
    ----------
    dt: float
        Sample distance in seconds
    nf: int
        Number of logarithmically spaced frequencies between fmin and fmax
    f_min: int, default = 1
        Minimum frequency for transformation
    f_max: int , default = 50
        Maximum frequency for transformation
    wl: str, default = 'morlet'
        Wavelet to use, Obspy, Scipy, and Pywt wavelets. Use list_wavelets() to see options.
    w0: int, default = 5
        parameter for the wavelet, tradeoff between time and frequency resolution
    widths: array_like, default = None
        The wavelet scales to use. For using this parameter. Wavelets other than morlet
        and mexh needs widths parameter.
    
    Returns
    -------
    Widths to use in transformation
    """
    _check_args(widths=widths, f_min=f_min, f_max=f_max, nf=nf)
    if widths is None:
        sampling_freq = 1/dt
        freq = np.logspace(np.log10(f_min), np.log10(f_max), nf)
        if wl in ['ricker','mexh']:
            widths = 1/(4*freq)

        elif wl in ['morl', 'morlet_s']:                         # Morlet wavelet has a 'widths' function that is 
            widths = w0*sampling_freq / (2*freq*np.pi)           # calculated using frequencies, sampling_frequency
                                                                 # and wavelet parameter. You can find this function at 
                                                                 # the usage example of cwt from scipy
                                                                 # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.morlet2.html
    else:
        pass
    
    return widths



def calc_parameter(wl, dt, f_min, f_max, nf, w0, widths):
    """
    Parameter Calculator for different wavelet functions
    
    Parameters
    ----------
    wl: str, default = 'morlet'
        Wavelet to use, Obspy, Scipy, and Pywt wavelets. Use list_wavelets() to see options.
    dt: float
        Sample distance in seconds
    nf: int
        Number of logarithmically spaced frequencies between fmin and fmax
    f_min: int, default=1
        Minimum frequency for transformation
    f_max: int , default = 50
        Maximum frequency for transformation
    w0: int, default = 5
        parameter for the wavelet, tradeoff between time and frequency resolution
    widths: array_like, default = None
        The wavelet scales to use. Wavelets other than morlet
        and mexh need widths parameter.
     
    
    Returns
    -------
    Parameters as dictionary : params
    """
    _check_args(widths=widths, f_min=f_min, f_max=f_max, nf=nf)
    
    if wl == 'ricker':									# Scipy				
        assert w0 is None
        widths = _calc_widths(dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0, widths=widths, wl=wl)
        params = {'widths':widths}
    elif wl == 'mexh':									# Pywt
        assert w0 is None
        widths = _calc_widths(dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0, widths=widths, wl=wl)
        params = {'scales':widths}
    elif wl == 'morlet_s':								# Scipy			
        widths = _calc_widths(dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0, widths=widths, wl=wl)
        params = {'widths':widths,'w':w0}
    elif wl =='morlet':								# Obspy
        params = {'dt':dt, 'fmin':f_min, 'fmax':f_max, 'nf':nf, 'w0':w0}
    elif wl == 'morl':
        widths = _calc_widths(dt=dt, f_min=f_min, f_max=f_max, nf=nf, w0=w0, widths=widths, wl=wl)
        params = {'scales':widths}					# Pywt
    else:
        if widths is None:
            raise ValueError('This wavelet can be used with widths')
        params = {'scales':widths}					# Pywt
    return params
    


def _check_args(widths, f_min, f_max, nf):
    
    if widths is None:
        assert (f_min is not None) and (f_max is not None) and (nf is not None)
    else:
        assert (f_min is None) and (f_max is None) and (nf is None)
