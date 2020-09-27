def repr_(self_types):
    result_types = set()
    for ty in self_types:
        if hasattr(ty, 'mro'):
            result_types.add(str)
        else:
            result_types.add(repr(ty))
    return result_types
