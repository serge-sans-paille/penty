from penty.types import Cst as _Cst, Module as _Module, Type as _Ty
from penty.types import astype as _astype
import typing as _typing
import operator as _operator

##
#

def bool_not(self_types):
    result_types = set()
    for o in self_types:
        if o is bool:
            result_types.add(bool)
        elif issubclass(o, _Cst):
            result_types.add(_Cst[not o.__args__[0]])
        else:
            raise NotImplementedError
    return result_types

_bool_attrs = {
    '__not__' : bool_not,
}

##
#

def int_init(self_types):
    result_types = set()
    for ty in self_types:
        if ty in (bool, int, float, str):
            result_types.add(int)
        elif issubclass(ty, _Cst):
            result_types.add(_Cst[int(ty.__args__[0])])
        else:
            raise NotImplementedError
    return result_types


def int_make_binop(operator):
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

def int_make_bitop(operator):
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

def int_truediv(self_types, other_types):
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

def int_make_unaryop(operator):
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

def int_boolean(self_types):
    result_types = set()
    for o in self_types:
        if o is int:
            result_types.add(bool)
        elif issubclass(o, _Cst):
            result_types.add(_Cst[bool(o.__args__[0])])
        else:
            raise NotImplementedError
    return result_types

def int_make_boolop(operator):
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

def int_make_biniop(operator):
    def biniop(self_types, other_types):
        result_types = operator(self_types, other_types)
        # int are immutable so we don't update self_types
        return result_types
    return biniop


_int_attrs = {
    '__add__': int_make_binop(_operator.add),
    '__and__': int_make_bitop(_operator.and_),
    '__bool__': int_boolean,
    '__eq__': int_make_boolop(_operator.eq),
    '__floordiv__': int_make_binop(_operator.floordiv),
    '__ge__': int_make_boolop(_operator.ge),
    '__gt__': int_make_boolop(_operator.gt),
    '__init__': _Cst[int_init],
    '__invert__': int_make_unaryop(_operator.inv),
    '__le__': int_make_boolop(_operator.le),
    '__lt__': int_make_boolop(_operator.lt),
    '__mul__': int_make_binop(_operator.mul),
    '__mod__': int_make_binop(_operator.mod),
    '__ne__': int_make_boolop(_operator.ne),
    '__neg__': int_make_unaryop(_operator.neg),
    '__or__': int_make_bitop(_operator.or_),
    '__pos__': int_make_unaryop(_operator.pos),
    '__pow__': int_make_binop(_operator.pow),
    '__sub__': int_make_binop(_operator.sub),
    '__truediv__': int_truediv,
    '__xor__': int_make_bitop(_operator.xor),
}
_int_attrs.update({
    '__iadd__': int_make_biniop(_int_attrs['__add__']),
    '__iand__': int_make_biniop(_int_attrs['__and__']),
    '__ior__': int_make_biniop(_int_attrs['__or__']),
    '__itruediv__': int_make_biniop(_int_attrs['__truediv__']),
    '__ifloordiv__': int_make_biniop(_int_attrs['__floordiv__']),
    '__imod__': int_make_biniop(_int_attrs['__mod__']),
    '__imul__': int_make_biniop(_int_attrs['__mul__']),
    '__isub__': int_make_biniop(_int_attrs['__sub__']),
    '__ipow__': int_make_biniop(_int_attrs['__pow__']),
    '__ixor__': int_make_biniop(_int_attrs['__xor__']),
})

##
#

_float_attrs = {
}

##
#
str_iterator = type(iter(""))

def str_iter(self_types):
    result_types = set()
    for o in self_types:
        if o is str:
            result_types.add(str_iterator)
        elif issubclass(o, _Cst):
            result_types.add(str_iterator)
        else:
            raise NotImplementedError
    return result_types

_str_attrs = {
    '__iter__' : str_iter,
}

##
#

def dict_clear(base_ty, self_types):
    for self_ty in self_types:
        if self_ty is not base_ty:
            raise NotImplementedError
    return {_Cst[None]}

def dict_fromkeys(iterable_types, value_types=None):
    from penty.penty import Types

    iter_types = set()
    for ty in iterable_types:
        iter_types.update(Types[ty]['__iter__'](iterable_types))

    key_types = set()
    for ty in iter_types:
        key_types.update(Types[ty]['__next__'](iter_types))

    if value_types is None:
        return {_typing.Dict[k, _Cst[None]] for k in key_types}
    else:
        return {_typing.Dict[k, v]
                for k in key_types
                for v in value_types}

def dict_instanciate(ty):
    return {
        'clear': lambda *args: dict_clear(ty, *args),
    }

_dict_attrs = {
    'from_keys': _Cst[dict_fromkeys],
}

##
#

def list_count(base_ty, self_types, elt_types):
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

def list_getitem(base_ty, self_types, key_types):
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

def list_instanciate(ty):
    return {
        '__getitem__': lambda *args: list_getitem(ty, *args),
        'count': lambda *args: list_count(ty, *args),
    }

##
#

def tuple_getitem(base_ty, self_types, key_types):
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


def tuple_instanciate(ty):
    return {
        '__getitem__': lambda *args: tuple_getitem(ty, *args),
    }

##
#

def str_iterator_next(self_types):
    result_types = set()
    for o in self_types:
        if o is str_iterator:
            result_types.add(str)
        elif isinstance(o, str_iterator):
            result_types.add(str)
        else:
            raise NotImplementedError
    return result_types

_str_iterator_attrs = {
    '__next__' : str_iterator_next,
}

##
#

def id_(self_types):
    return {int}


##
#

def repr_(self_types):
    result_types = set()
    for ty in self_types:
        if issubclass(ty, _Cst):
            ty_v = ty.__args__[0]
            result_types.add(_Cst[repr(ty_v)])
        else:
            result_types.add(str)
    return result_types

##
#

def slice_(lower_types, upper_types, step_types):
    if all(len(tys) == 1 for tys in (lower_types, upper_types, step_types)):
        lower_ty = next(iter(lower_types))
        upper_ty = next(iter(upper_types))
        step_ty = next(iter(step_types))

        isstatic = all(issubclass(ty, _Cst)
                       for ty in (lower_ty, upper_ty, step_ty))

        if isstatic:
            return {_Cst[slice(*(ty.__args__[0]
                                 for ty in (lower_ty, upper_ty, step_ty)))]}

    return {slice}

##
#

def type_(self_types):
    from penty.penty import Types
    return {_Ty[_astype(ty)] for ty in self_types}

##
#

def register(registry):
    if _Module['builtins'] not in registry:
        registry[bool] = _bool_attrs
        registry[dict] = _dict_attrs
        registry[int] = _int_attrs
        registry[float] = _float_attrs
        registry[str] = _str_attrs
        registry[str_iterator] = _str_iterator_attrs
        registry[_typing.Dict] = dict_instanciate
        registry[_typing.List] = list_instanciate
        registry[_typing.Tuple] = tuple_instanciate

        registry[_Module['builtins']] = {
            'dict': {_Ty[dict]},
            'id': {_Cst[id_]},
            'int': {_Ty[int]},
            'repr': {_Cst[repr_]},
            'slice': {_Cst[slice_]},
            'type': {_Cst[type_]},
        }
