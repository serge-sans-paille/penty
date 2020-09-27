from penty.types import Cst as _Cst

def repr_(self_types):
    result_types = set()
    for ty in self_types:
        if issubclass(ty, _Cst):
            ty_v = ty.__args__[0]
            result_types.add(_Cst[repr(ty_v)])
        else:
            result_types.add(str)
    return result_types
