def add(self_types, other_types):
    result_types = set()
    for o in other_types:
        if o in (bool, int):
            result_types.add(int)
        elif o is float:
            result_types.add(float)
        else:
            raise NotImplementedError
    return result_types

mul = sub = matmul = mod = floordiv = pow = add

def bitor(self_types, other_types):
    result_types = set()
    for o in other_types:
        if o in (bool, int):
            result_types.add(int)
        else:
            raise NotImplementedError
    return result_types

bitand = bitxor = bitor

def truediv(self_types, other_types):
    result_types = set()
    for o in other_types:
        if o in (bool, int, float):
            result_types.add(float)
        else:
            raise NotImplementedError
    return result_types

def pos(self_types):
    assert self_types == {int}
    return {int}

neg = invert = pos
