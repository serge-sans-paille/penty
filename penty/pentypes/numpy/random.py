from penty.types import Cst as _Cst, Module as _Module, astype as _astype

def bytes_(int_types):
    result_types = set()
    for s in int_types:
        s = _astype(s)
        if s is int:
            result_types.add(str)
        else:
            raise NotImplementedError
    return result_types

def register(registry):
    if _Module['numpy.random'] not in registry:
        registry[_Module['numpy.random']] = {
            'bytes': _Cst[bytes_],
        }
