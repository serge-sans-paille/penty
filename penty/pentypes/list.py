from penty.pentypes.builtins.slice import frozen_slice as _frozen_slice

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
            elif isinstance(key_ty, (bool, int)):
                result_types.add(base_key_ty)
            elif key_ty is slice:
                result_types.add(base_ty)
            elif isinstance(key_ty, _frozen_slice):
                result_types.add(base_ty)
            else:
                raise NotImplementedError

    return result_types
