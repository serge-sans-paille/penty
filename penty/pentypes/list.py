from penty.types import Cst as _Cst

def instanciate(ty):
    return {
        '__getitem__': lambda *args: getitem(ty, *args),
    }

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
