def not_(self_types):
    result_types = set()
    for o in self_types:
        if o is bool:
            result_types.add(bool)
        elif isinstance(o, bool):
            result_types.add(not o)
        else:
            raise NotImplementedError
    return result_types
