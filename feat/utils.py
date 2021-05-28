import numpy as np
import pywt


 
def list_wavelets()
    """"
    Returns available wavelets.
    
    """
    wavlist = ['morlet', 'ricker','morlet_s'] + pywt.wavelist(kind='continuous')
    return(wavlist)
