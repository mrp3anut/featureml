import numpy as np
import pywt 
from numpy.testing import (assert_allclose, assert_equal)
from feat.core import cwt, feautirize_cwt


def test_cwt_dim():
    data = np.zeros((600,))
    dt = 0.01
    nf = 1
    f_min = 1
    f_max = 2
    wl = 'morlet'
    w0=5
    assert cwt(data = data, dt=dt,nf=nf,f_min=f_min,f_max=f_max,wl=wl,w0=w0).ndim == 2


def test_cwt_shape():
    data = np.zeros((600,))
    dt = 0.01
    nf = 2
    f_min = 1
    f_max = 2
    wl = 'morlet'
    w0 =5
    assert cwt(data = data, dt=dt,nf=nf,f_min=f_min,f_max=f_max,wl=wl).shape == (600,2)
    
def test_featurize_cwt_shape():
    
    data = np.zeros((600,))
    dt = 0.01
    nf = 2
    f_min = 1
    f_max = 2
    wl = 'morlet'
    w0 =5
    assert featutize_cwt(data = data, dt=dt,nf=nf,f_min=f_min,f_max=f_max,wl=wl,w0=w0).shape == (600,3) 
    
    

def test_cwt_advanced(tol=1e-7):
    time, sst = pywt.data.nino()  
    # Nino is a small sea surface temperature dataset
    sst = np.asarray(sst)
    dt = time[1] - time[0] #dt is time difference between two samples
    
    wavelet = 'morlet'
    f_min=1
    f_max=10
    nf=1
    w0=5
    
    cfs  = np.reshape(cwt(data = data, dt=dt,nf=nf,f_min=f_min,f_max=f_max,wl=wl,w0=w0),(264))  # We take cwt of the dataset, 
    										         #then change its dimention from (264,1) to (264,)
    assert_equal(cfs.real.dtype, sst.dtype)  

    sst_complex = sst + 1j*sst                 #We create a new complex data
    cfs_complex,  = cwt(sst_complex, wl=wavelet,f_min=f_min,f_max=f_max,dt=dt) #Then we take cwt again on complex data 
    
    
    assert_allclose(cfs + 1j*cfs, cfs_complex, atol=tol, rtol=tol) #complex valued transform equals to sum of 
    								                               #the transforms of the real and imaginary components


