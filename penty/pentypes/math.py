from penty.types import Cst as _Cst, Module as _Module, Type as _Ty
from penty.types import ConstFunctionType as _CFT, Tuple as _Tuple

import math as _math

##
#

def ceil_(self_type):
    from penty.penty import Types
    if issubclass(self_type, float):
        return float
    return Types[self_type]['__ceil__'](self_type)

##
#

def register(registry):
    if _Module['math'] not in registry:

        registry[_Module['math']] = {
            'ceil': {_CFT[ceil_, _math.ceil]},
        }
