import numpy as np
from numpy.testing import (assert_allclose, assert_equal)
import pywt 
from feat.core import cwt, featurize_cwt



F_MIN = 1
F_MAX = 10
W0 = 5


def test_cwt_shape_morl():
    data = np.zeros((600,))
    dt = 0.01
    nf = 2
    wavelets = ['morlet', 'morlet_s', 'morl']
    for wl in wavelets:
    	assert cwt(data=data, dt=dt, nf=nf, f_min=F_MIN, f_max=F_MAX, wl=wl, w0=W0).shape == (600,2) 
    	
def test_cwt_shape_mexh():
    data = np.ones((600,))
    widths = np.arange(1,31)
    dt=0.01
    wl='ricker'
    assert cwt(data=data,dt=dt, widths=widths, wl=wl).shape == (600,30) 
    	    	
    
def test_featurize_cwt_shape():
    
    data = np.zeros((600,))
    dt = 0.01
    nf = 2
    wl = 'morlet'
    
    wavelets = ['morlet', 'morlet_s', 'morl']
    for wl in wavelets:
    	assert featurize_cwt(data=data, dt=dt, nf=nf, f_min=F_MIN, f_max=F_MAX, wl=wl, w0=W0).shape == (600,3) 
    
    

def test_complex_linearity(tol=1e-7):
    time, sst = pywt.data.nino()  
    # Nino is a small sea surface temperature dataset
    data = np.asarray(sst)
    dt = time[1] - time[0] # dt is time difference between two samples
    
    wl = 'morlet'
    nf=1

    
    cfs  = cwt(data=data, dt=dt, nf=nf, f_min=F_MIN, f_max=F_MAX, wl=wl, w0=W0)  # We take cwt of the dataset, 
    										        
    assert_equal(cfs.real.dtype, sst.dtype)  

    sst_complex = sst + 1j*sst                 # We create a new complex data
    cfs_complex = cwt(sst_complex, dt=dt, nf=nf, f_min=F_MIN, f_max=F_MAX, wl=wl, w0=W0) # Then we take cwt again on complex data 
    
    
    assert_allclose(cfs + 1j*cfs, cfs_complex, atol=tol, rtol=tol) # Complex valued transform equals to sum of 
    								     # The transforms of the real and imaginary components


# This function tests the extended data, since extended data is data with cwt transform, one part of the data must be equal to it self
# and other part must be equal to the cwt of the data 
def test_featurize_extend(tol=1e-7):
    time, sst = pywt.data.nino()  
   
    data = np.asarray(sst)
    dt = time[1] - time[0] 
    
    wl='morlet'
    f_min=1
    f_max=10
    nf=1
    w0=5
    extended = featurize_cwt(data=data, dt=dt, nf=nf, f_min=F_MIN, f_max=F_MAX, wl=wl,w0=W0)
    cwts = cwt(data=data, dt=dt, nf=nf, f_min=F_MIN, f_max=F_MAX, wl=wl,w0=W0)
    assert_allclose(sst, extended[:,0], atol=tol, rtol=tol) # Check that first channel of the extended data equals to original data
    assert_allclose(cwts[:,0], extended[:,1], atol=tol, rtol=tol)
