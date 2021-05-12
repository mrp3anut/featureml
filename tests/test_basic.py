import numpy as np
from feat.core import cwt



def test_dim():
    data = np.zeros((600,))
    dt = 0.01
    nf = 1
    f_min = 1
    f_max = 2
    wl = 'morlet'
    assert cwt(data = data, dt=dt,nf=nf,f_min=f_min,f_max=f_max,wl=wl).ndim == 2


