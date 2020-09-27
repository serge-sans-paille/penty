from penty.types import Cst as _Cst

def instanciate(ty):
    return {
        '__getitem__': lambda *args: getitem(ty, *args),
        'count': lambda *args: count(ty, *args),
    }

def count(base_ty, self_types, elt_types):
    base_key_ty, = base_ty.__args__
    result_types = set()
    for self_ty in self_types:
        if self_ty is not base_ty:
            raise NotImplementedError
        for elt_ty in elt_types:
            if elt_ty is base_key_ty:
                result_types.add(int)
            elif issubclass(elt_ty, _Cst):
                elt_v = elt_ty.__args__[0]
                if type(elt_v) is base_key_ty:
                    result_types.add(int)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
    return result_types

def getitem(base_ty, self_types, key_types):
    base_key_ty, = base_ty.__args__

    result_types = set()
    for self_ty in self_types:
        if self_ty is not base_ty:
            raise NotImplementedError
        for key_ty in key_types:
            if key_ty in (bool, int):
                result_types.add(base_key_ty)
            elif key_ty is slice:
                result_types.add(base_ty)
            elif issubclass(key_ty, _Cst):
                key_v = key_ty.__args__[0]
                if isinstance(key_v, (bool, int)):
                    result_types.add(base_key_ty)
                elif isinstance(key_v, slice):
                    result_types.add(base_ty)
            else:
                raise NotImplementedError

    return result_types
