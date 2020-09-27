from penty.types import Cst as _Cst

def not_(self_types):
    result_types = set()
    for o in self_types:
        if o is bool:
            result_types.add(bool)
        elif issubclass(o, _Cst):
            result_types.add(_Cst[not o.__args__[0]])
        else:
            raise NotImplementedError
    return result_types
