import numpy as np 
import pySZe

def test_par():
	par = pySZe.par()
	assert np.abs(par.cosmo.Om-0.315)<1e-2