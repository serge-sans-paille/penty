import operator as _operator

def _make_binop(operator):
    def binop(self_types, other_types):
        result_types = set()
        for s in self_types:
            if isinstance(s, int):
                for o in other_types:
                    if o in (bool, int):
                        result_types.add(int)
                    elif o is float:
                        result_types.add(float)
                    elif isinstance(o, int):
                        result_types.add(operator(s, o))
                    else:
                        raise NotImplementedError
            elif s is int:
                for o in other_types:
                    if o in (bool, int):
                        result_types.add(int)
                    elif o is float:
                        result_types.add(float)
                    elif isinstance(o, int):
                        result_types.add(int)
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError
        return result_types
    return binop

add = _make_binop(_operator.add)
mul = _make_binop(_operator.mul)
sub = _make_binop(_operator.sub)
matmul = _make_binop(_operator.matmul)
mod = _make_binop(_operator.mod)
floordiv = _make_binop(_operator.floordiv)
power = _make_binop(_operator.pow)

def _make_bitop(operator):
    def binop(self_types, other_types):
        result_types = set()
        for s in self_types:
            if isinstance(s, int):
                for o in other_types:
                    if o in (bool, int):
                        result_types.add(int)
                    elif isinstance(o, int):
                        result_types.add(operator(s, o))
                    else:
                        raise NotImplementedError
            elif s is int:
                for o in other_types:
                    if o in (bool, int):
                        result_types.add(int)
                    elif isinstance(o, int):
                        result_types.add(int)
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError
        return result_types
    return binop

bitor = _make_bitop(_operator.or_)
bitand = _make_bitop(_operator.and_)
bitxor = _make_bitop(_operator.xor)

def truediv(self_types, other_types):
    result_types = set()
    for s in self_types:
        if isinstance(s, int):
            for o in other_types:
                if o in (bool, int, float):
                    result_types.add(float)
                elif isinstance(o, (bool, int, float)):
                    result_types.add(s / o)
                else:
                    raise NotImplementedError
        elif s is int:
            for o in other_types:
                if o in (bool, int, float):
                    result_types.add(float)
                elif isinstance(o, (bool, int, float)):
                    result_types.add(float)
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError
    return result_types

def _make_unaryop(operator):
    def unaryop(self_types):
        result_types = set()
        for o in self_types:
            if o is int:
                result_types.add(int)
            elif isinstance(o, int):
                result_types.add(operator(o))
            else:
                raise NotImplementedError
        return result_types
    return unaryop

pos = _make_unaryop(_operator.pos)
neg = _make_unaryop(_operator.neg)
invert = _make_unaryop(_operator.inv)

def boolean(self_types):
    result_types = set()
    for o in self_types:
        if o is int:
            result_types.add(bool)
        elif isinstance(o, int):
            result_types.add(bool(o))
        else:
            raise NotImplementedError
    return result_types
