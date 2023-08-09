'''
Created on 08 December 2020
@author: Sambit Giri
Setup script
'''

from .params import par
from .constants import *
from .profiles import profiles
from .cosmo import cosmo, rhoc_of_z

from .kSZ import *
from .tSZ import *
# from . import ACT
from . import cosmo
from . import profiles
from .telescopes import * 

from . import gertsenshtein_effect

#Suppress warnings from zero-divisions and nans
import numpy
numpy.seterr(all='ignore')