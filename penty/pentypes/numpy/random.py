from penty.types import Cst as _Cst, Module as _Module, astype as _astype

def bytes_(int_ty):
    int_ty = _astype(int_ty)
    if int_ty is int:
        return str
    else:
        raise NotImplementedError

def register(registry):
    if _Module['numpy.random'] not in registry:
        registry[_Module['numpy.random']] = {
            'bytes': _Cst[bytes_],
        }
