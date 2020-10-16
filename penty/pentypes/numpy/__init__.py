from penty.pentypes.numpy import random
from penty.types import Module as _Module

def register(registry):
    if _Module['numpy'] not in registry:
        registry[_Module['numpy']] = {
            'random': _Module['numpy.random'],
        }
