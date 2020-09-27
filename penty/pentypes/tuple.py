from penty.types import Cst as _Cst

import typing as _typing

def instanciate(ty):
    return {
        '__getitem__': lambda *args: getitem(ty, *args),
    }

def getitem(base_ty, self_types, key_types):
    base_key_types = base_ty.__args__

    result_types = set()
    for self_ty in self_types:
        if self_ty is not base_ty:
            raise NotImplementedError
        for key_ty in key_types:
            if key_ty in (bool, int):
                result_types.update(base_key_types)
            elif key_ty is slice:
                result_types.add(tuple)
            elif issubclass(key_ty, _Cst):
                key_v = key_ty.__args__[0]
                if isinstance(key_v, (bool, int)):
                    result_types.add(base_key_types[key_v])
                elif isinstance(key_v, slice):
                    result_types.add(_typing.Tuple[base_key_types[key_v]])
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    return result_types
