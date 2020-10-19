from penty.types import Cst as _Cst, Module as _Module, Type as _Ty
from penty.types import astype as _astype
import typing as _typing
import operator as _operator

##
#

def bool_bool(self_ty):
    return self_ty

def bool_init(value_ty):
    from penty.penty import Types
    return Types[_astype(value_ty)]['__bool__'](value_ty)

def bool_not(self_ty):
    if self_ty is bool:
        return bool
    elif issubclass(self_ty, _Cst):
        return _Cst[not self_ty.__args__[0]]
    else:
        raise TypeError

_bool_attrs = {
    '__bool__' : _Cst[bool_bool],
    '__init__': _Cst[bool_init],
    '__not__' : _Cst[bool_not],
}

##
#

def int_init(self_ty):
    if self_ty in (bool, int, float, str):
        return int
    elif issubclass(self_ty, _Cst):
        return _Cst[int(ty.__args__[0])]
    else:
        raise TypeError


def int_make_binop(operator):
    def binop(self_ty, other_ty):
        if issubclass(self_ty, _Cst):
            if other_ty in (bool, int):
                return int
            elif other_ty is float:
                return float
            elif issubclass(other_ty, _Cst):
                return _Cst[operator(self_ty.__args__[0],
                                     other_ty.__args__[0])]
            else:
                raise TypeError
        elif self_ty is int:
            if other_ty in (bool, int):
                return int
            elif other_ty is float:
                return float
            elif issubclass(other_ty, _Cst):
                return int
            else:
                raise TypeError
        else:
            raise TypeError
        return result_types
    return _Cst[binop]

def int_make_bitop(operator):
    def binop(self_ty, other_ty):
        if issubclass(self_ty, _Cst):
            if other_ty in (bool, int):
                return int
            elif issubclass(other_ty, _Cst):
                return _Cst[operator(self_ty.__args__[0],
                                     other_ty.__args__[0])]
            else:
                raise TypeError
        elif self_ty is int:
            if other_ty in (bool, int):
                return int
            elif isinstance(other_ty, int):
                return int
            else:
                raise TypeError
        else:
            raise TypeError
    return _Cst[binop]

def int_truediv(self_ty, other_ty):
    if issubclass(self_ty, _Cst):
        if other_ty in (bool, int, float):
            return float
        elif issubclass(other_ty, _Cst):
            return _Cst[self_ty.__args__[0] / other_ty.__args__[0]]
        else:
            raise TypeError
    elif self_ty is int:
        if other_ty in (bool, int, float):
            return float
        elif issubclass(other_ty, _Cst):
            return float
        else:
            raise TypeError
    else:
        raise TypeError

def int_make_unaryop(operator):
    def unaryop(self_ty):
        if self_ty is int:
            return int
        elif issubclass(self_ty, _Cst):
            return _Cst[operator(self_ty.__args__[0])]
        else:
            raise TypeError
    return _Cst[unaryop]

def int_boolean(self_ty):
    if self_ty is int:
        return bool
    elif issubclass(self_ty, _Cst):
        return _Cst[bool(self_ty.__args__[0])]
    else:
        raise TypeError

def int_make_boolop(operator):
    def boolop(self_ty, other_ty):
        if issubclass(self_ty, _Cst):
            if other_ty in (bool, int, float):
                return bool
            elif issubclass(other_ty, _Cst):
                return _Cst[operator(self_ty.__args__[0],
                                     other_ty.__args__[0])]
            else:
                raise TypeError
        elif self_ty is int:
            if other_ty in (bool, int, float):
                return bool
            elif issubclass(other_ty, _Cst):
                return bool
            else:
                raise TypeError
        else:
            raise TypeError
    return _Cst[boolop]

def int_make_biniop(operator):
    def biniop(self_ty, other_ty):
        result_ty = operator(self_ty, other_ty)
        # int are immutable so we don't update self_ty
        return result_ty
    return _Cst[biniop]


_int_attrs = {
    '__add__': int_make_binop(_operator.add),
    '__and__': int_make_bitop(_operator.and_),
    '__bool__': _Cst[int_boolean],
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
    '__truediv__': _Cst[int_truediv],
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

def float_bool(self_ty):
    if issubclass(self_ty, _Cst):
        return _Cst[bool(self_ty.__args__[0])]
    else:
        return bool


_float_attrs = {
    '__bool__' : _Cst[float_bool],
}

##
#
str_iterator = type(iter(""))

def str_bool(self_ty):
    if self_ty is str:
        return bool
    elif issubclass(self_ty, _Cst):
        return _Cst[bool(self_ty.__args__[0])]
    else:
        raise TypeError


def str_iter(self_ty):
    if self_ty is str:
        return str_iterator
    elif issubclass(self_ty, _Cst):
        return str_iterator
    else:
        raise TypeError

_str_attrs = {
    '__bool__': _Cst[str_bool],
    '__iter__' : _Cst[str_iter],
    '__len__': _Cst[lambda *args: int],
}

##
#

def dict_clear(base_ty, self_ty):
    if self_ty is not base_ty:
        raise TypeError
    return _Cst[None]

def dict_fromkeys(iterable_ty, value_ty=None):
    from penty.penty import Types

    iter_ty = Types[iterable_ty]['__iter__'](iterable_ty)
    key_ty = Types[iter_ty]['__next__'](iter_ty)

    if value_ty is None:
        return _typing.Dict[key_ty, _Cst[None]]
    else:
        return _typing.Dict[key_ty, value_ty]

def dict_instanciate(ty):
    return {
        '__bool__': _Cst[lambda *args: bool],
        '__len__': _Cst[lambda *args: int],
        'clear': _Cst[lambda *args: dict_clear(ty, *args)],
    }

_dict_attrs = {
    'clear': _Cst[lambda x: dict_clear(x, x)],
    'from_keys': _Cst[dict_fromkeys],
}

##
#

def list_append(base_ty, self_ty, value_ty):
    return _Cst[None], (_typing.List[value_ty], value_ty)

def list_count(base_ty, self_ty, elt_ty):
    base_key_ty, = base_ty.__args__
    if self_ty is not base_ty:
        raise TypeError
    if elt_ty is base_key_ty:
        return int
    elif issubclass(elt_ty, _Cst):
        elt_v = elt_ty.__args__[0]
        if type(elt_v) is base_key_ty:
            return int
        else:
            raise TypeError
    else:
        raise TypeError

def list_getitem(base_ty, self_ty, key_ty):
    base_value_ty, = base_ty.__args__

    if self_ty is not base_ty:
        raise TypeError
    if key_ty in (bool, int):
        return base_value_ty
    elif key_ty is slice:
        return base_ty
    elif issubclass(key_ty, _Cst):
        key_v = key_ty.__args__[0]
        if isinstance(key_v, (bool, int)):
            return base_value_ty
        elif isinstance(key_v, slice):
            return base_ty
    else:
        raise TypeError

def list_instanciate(ty):
    return {
        '__bool__': _Cst[lambda *args: bool],
        '__getitem__': _Cst[lambda *args: list_getitem(ty, *args)],
        '__len__': _Cst[lambda *args: int],
        'append': _Cst[lambda *args: list_append(ty, *args)],
        'count': _Cst[lambda *args: list_count(ty, *args)],
    }

_list_attrs = {
    'append': _Cst[lambda self, elt: list_append(self, self, elt)],
}

##
#

def set_instanciate(ty):
    return {
        '__bool__': _Cst[lambda *args: bool],
        '__len__': _Cst[lambda *args: int],
    }

_set_attrs = {
}

##
#

def tuple_bool(base_ty, self_ty):
    return _Cst[bool(base_ty.__args__)]

def tuple_getitem(base_ty, self_ty, key_ty):
    base_value_types = base_ty.__args__

    if self_ty is not base_ty:
        raise TypeError
    if key_ty in (bool, int):
        return set(base_value_types)
    elif key_ty is slice:
        return tuple
    elif issubclass(key_ty, _Cst):
        key_v = key_ty.__args__[0]
        if isinstance(key_v, (bool, int)):
            return base_value_types[key_v]
        elif isinstance(key_v, slice):
            return _typing.Tuple[base_value_types[key_v]]
        else:
            raise TypeError
    else:
        raise TypeError


def tuple_instanciate(ty):
    return {
        '__bool__': _Cst[lambda *args: tuple_bool(ty, *args)],
        '__getitem__': _Cst[lambda *args: tuple_getitem(ty, *args)],
        '__len__': _Cst[lambda *args: _Cst[len(ty.__args__)]],
    }

##
#

def str_iterator_next(self_ty):
    if self_ty is str_iterator:
        return str
    elif isinstance(self_ty, str_iterator):
        return str
    else:
        raise TypeError

_str_iterator_attrs = {
    '__next__' : _Cst[str_iterator_next],
}

##
#

def id_(self_types):
    return int

##
#

def repr_(self_ty):
    if issubclass(self_ty, _Cst):
        self_v = self_ty.__args__[0]
        return _Cst[repr(self_v)]
    else:
        return str

##
#

def len_(self_type):
    from penty.penty import Types
    if issubclass(self_type, _Cst):
        return _Cst[len(self_type.__args__[0])]
    else:
        return Types[self_type]['__len__'](self_type)

##
#

def slice_(lower_ty, upper_ty, step_ty):
    isstatic = all(issubclass(ty, _Cst)
                   for ty in (lower_ty, upper_ty, step_ty))
    if isstatic:
        return _Cst[slice(*(ty.__args__[0]
                            for ty in (lower_ty, upper_ty, step_ty)))]
    return slice

##
#

def type_(self_ty):
    from penty.penty import Types
    return _Ty[_astype(self_ty)]

##
#

def register(registry):
    if _Module['builtins'] not in registry:
        registry[bool] = _bool_attrs
        registry[dict] = _dict_attrs
        registry[list] = _list_attrs
        registry[float] = _float_attrs
        registry[int] = _int_attrs
        registry[set] = _set_attrs
        registry[str] = _str_attrs
        registry[str_iterator] = _str_iterator_attrs
        registry[_typing.Dict] = dict_instanciate
        registry[_typing.List] = list_instanciate
        registry[_typing.Set] = set_instanciate
        registry[_typing.Tuple] = tuple_instanciate

        registry[_Module['builtins']] = {
            'bool': {_Ty[bool]},
            'dict': {_Ty[dict]},
            'id': {_Cst[id_]},
            'int': {_Ty[int]},
            'len': {_Cst[len_]},
            'repr': {_Cst[repr_]},
            'slice': {_Cst[slice_]},
            'type': {_Cst[type_]},
        }
