import numpy as np
import matplotlib.pyplot as plt
import pywt


def plt_ch(ext_data):
    """
    Plots given data 
    
    Parameters
    ------
    ext_data: 2D Array
    
    Returns
    -------
    Plot 
    """ 
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
