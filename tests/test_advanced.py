import numpy as np
import pywt 
from numpy.testing import (assert_allclose, assert_equal)
from feat.core import cwt 




def test_cwt_advanced(tol=1e-7):
    time, sst = pywt.data.nino()
    sst = np.asarray(sst)
    dt = time[1] - time[0]
    wavelet = 'morlet'
    f_min=1
    f_max=10

    
    cfs  = np.reshape(cwt(sst, wl=wavelet, f_min=f_min, f_max=f_max, dt=dt),(264,))
    assert_equal(cfs.real.dtype, sst.dtype)

    sst_complex = sst + 1j*sst
    cfs_complex,  = cwt(sst_complex, wl=wavelet, f_min=f_min, f_max=f_max, dt=dt)
    assert_allclose(cfs + 1j*cfs, cfs_complex, atol=tol, rtol=tol)
