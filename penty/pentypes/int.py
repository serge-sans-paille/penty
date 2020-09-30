import operator as _operator
from penty.types import Cst as _Cst

def _make_binop(operator):
    def binop(self_types, other_types):
        result_types = set()
        for s in self_types:
            if issubclass(s, _Cst):
                for o in other_types:
                    if o in (bool, int):
                        result_types.add(int)
                    elif o is float:
                        result_types.add(float)
                    elif issubclass(o, _Cst):
                        result_types.add(_Cst[operator(s.__args__[0],
                                                       o.__args__[0])])
                    else:
                        raise NotImplementedError
            elif s is int:
                for o in other_types:
                    if o in (bool, int):
                        result_types.add(int)
                    elif o is float:
                        result_types.add(float)
                    elif issubclass(o, _Cst):
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
            if issubclass(s, _Cst):
                for o in other_types:
                    if o in (bool, int):
                        result_types.add(int)
                    elif issubclass(o, _Cst):
                        result_types.add(_Cst[operator(s.__args__[0],
                                                       o.__args__[0])])
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
        if issubclass(s, _Cst):
            for o in other_types:
                if o in (bool, int, float):
                    result_types.add(float)
                elif issubclass(o, _Cst):
                    result_types.add(_Cst[s.__args__[0] / o.__args__[0]])
                else:
                    raise NotImplementedError
        elif s is int:
            for o in other_types:
                if o in (bool, int, float):
                    result_types.add(float)
                elif issubclass(o, _Cst):
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
            elif issubclass(o, _Cst):
                result_types.add(_Cst[operator(o.__args__[0])])
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
        elif issubclass(o, _Cst):
            result_types.add(_Cst[bool(o.__args__[0])])
        else:
            raise NotImplementedError
    return result_types

def _make_boolop(operator):
    def boolop(self_types, other_types):
        result_types = set()
        for s in self_types:
            if issubclass(s, _Cst):
                for o in other_types:
                    if o in (bool, int, float):
                        result_types.add(bool)
                    elif issubclass(o, _Cst):
                        result_types.add(_Cst[operator(s.__args__[0],
                                                       o.__args__[0])])
                    else:
                        raise NotImplementedError
            elif s is int:
                for o in other_types:
                    if o in (bool, int, float):
                        result_types.add(bool)
                    elif issubclass(o, _Cst):
                        result_types.add(bool)
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError
        return result_types
    return boolop

eq = _make_boolop(_operator.eq)
ge = _make_boolop(_operator.ge)
gt = _make_boolop(_operator.gt)
le = _make_boolop(_operator.le)
lt = _make_boolop(_operator.lt)
ne = _make_boolop(_operator.ne)

def _make_biniop(operator):
    def biniop(self_types, other_types):
        result_types = operator(self_types, other_types)
        # int are immutable so we don't update self_types
        return result_types
    return biniop

iadd = _make_biniop(add)
iand = _make_biniop(bitand)
ior = _make_biniop(bitor)
ixor = _make_biniop(bitxor)
itruediv = _make_biniop(truediv)
ifloordiv = _make_biniop(floordiv)
imod = _make_biniop(mod)
imul = _make_biniop(mul)
ipow = _make_biniop(power)
isub = _make_biniop(sub)
