'''
Created on 08 December 2020
@author: Sambit Giri
Setup script
'''

import sys 

from .params import par
from .kSZ import *
from .tSZ import *

# from . import ACT
from . import cosmo
from . import profiles
from .telescopes import * 

#Suppress warnings from zero-divisions and nans
import numpy
numpy.seterr(all='ignore')

# try:
#     __import__('pkg_resources').declare_namespace(__name__)
# except ImportError:
#     __path__ = __import__('pkgutil').extend_path(__path__, __name__)
