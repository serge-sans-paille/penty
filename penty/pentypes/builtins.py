from penty.types import Cst as _Cst, Module as _Module, Type as _Ty
from penty.types import astype as _astype, TypeOf as _TypeOf
from penty.types import ConstFunctionType as _CFT, Tuple as _Tuple
from penty.types import FunctionType as _FT
from penty.types import FilteringBool as _FilteringBool
from penty.types import List as _List, Set as _Set, Dict as _Dict
import operator as _operator

##
#

def bool_abs(value_ty):
    return int

def bool_bool(self_ty):
    return self_ty

def bool_int(value_ty):
    return int

def bool_float(value_ty):
    return float

def bool_str(value_ty):
    return str

def bool_init(value_ty):
    from penty.penty import Types
    return Types[_astype(value_ty)]['__bool__'](value_ty)

def bool_not(self_ty):
    if self_ty is bool:
        return bool
    else:
        raise TypeError

_bool_attrs = {
    '__abs__': _CFT[bool_abs, bool.__abs__],
    '__bool__': _CFT[bool_bool, bool.__bool__],
    '__float__': _CFT[bool_float, bool.__float__],
    '__init__': _CFT[bool_init, bool],
    '__int__': _CFT[bool_int, bool.__int__],
    '__not__': _CFT[bool_not, _operator.not_],
    '__str__': _CFT[bool_str, bool.__str__],
}

##
#

def FilteringBool_bool(self_ty):
    return _Cst[self_ty.__args__[0]]

def FilteringBool_not(self_ty):
    val, id_, filtered_tys = self_ty.__args__
    return _FilteringBool[
        not val,
        id_,
        filtered_tys]

_FilteringBool_attrs = {
    '__bool__': _FT[FilteringBool_bool],
    '__not__': _FT[FilteringBool_not],
}

##
#

def int_init(self_ty):
    from penty.penty import Types
    return Types[self_ty]['__int__'](self_ty)

def int_make_binop(operator):
    def binop(self_ty, other_ty):
        self_ty, other_ty = _astype(self_ty), _astype(other_ty)
        if self_ty is int:
            if other_ty in (bool, int):
                return int
            elif other_ty is float:
                return float
            else:
                raise TypeError
        else:
            raise TypeError
        return result_types
    return _CFT[binop, operator]

def int_make_bitop(operator):
    def binop(self_ty, other_ty):
        self_ty, other_ty = _astype(self_ty), _astype(other_ty)
        if self_ty is int:
            if other_ty in (bool, int):
                return int
            elif isinstance(other_ty, int):
                return int
            else:
                raise TypeError
        else:
            raise TypeError
    return _CFT[binop, operator]

def int_truediv(self_ty, other_ty):
    self_ty, other_ty = _astype(self_ty), _astype(other_ty)
    if self_ty is int:
        if other_ty in (bool, int, float):
            return float
        else:
            raise TypeError
    else:
        raise TypeError

def int_make_unaryop(operator):
    def unaryop(self_ty):
        if self_ty is int:
            return int
        else:
            raise TypeError
    return _CFT[unaryop, operator]

def int_bool(self_ty):
    if self_ty is int:
        return bool
    else:
        raise TypeError

def int_int(self_ty):
    if self_ty is int:
        return int
    else:
        raise TypeError

def int_abs(self_ty):
    if self_ty is int:
        return int
    else:
        raise TypeError

def int_float(self_ty):
    if self_ty is int:
        return float
    else:
        raise TypeError

def int_str(self_ty):
    if self_ty is int:
        return str
    else:
        raise TypeError

def int_make_boolop(operator):
    def boolop(self_ty, other_ty):
        self_ty, other_ty = _astype(self_ty), _astype(other_ty)
        if self_ty is int:
            if other_ty in (bool, int, float):
                return bool
            else:
                raise TypeError
        else:
            raise TypeError
    return _CFT[boolop, operator]

def int_make_biniop(operator):
    def biniop(self_ty, other_ty):
        result_ty = operator(self_ty, other_ty)
        # int are immutable so we don't update self_ty
        return result_ty
    return _CFT[biniop, operator]


_int_attrs = {
    '__abs__': _CFT[int_abs, int.__abs__],
    '__add__': int_make_binop(_operator.add),
    '__and__': int_make_bitop(_operator.and_),
    '__bool__': _CFT[int_bool, int.__bool__],
    '__eq__': int_make_boolop(_operator.eq),
    '__float__': _CFT[int_float, int.__float__],
    '__floordiv__': int_make_binop(_operator.floordiv),
    '__ge__': int_make_boolop(_operator.ge),
    '__gt__': int_make_boolop(_operator.gt),
    '__int__': _CFT[int_int, int],
    '__init__': _CFT[int_init, int],
    '__invert__': int_make_unaryop(_operator.inv),
    '__le__': int_make_boolop(_operator.le),
    '__lshift__': int_make_bitop(_operator.lshift),
    '__lt__': int_make_boolop(_operator.lt),
    '__mul__': int_make_binop(_operator.mul),
    '__mod__': int_make_binop(_operator.mod),
    '__ne__': int_make_boolop(_operator.ne),
    '__neg__': int_make_unaryop(_operator.neg),
    '__or__': int_make_bitop(_operator.or_),
    '__pos__': int_make_unaryop(_operator.pos),
    '__pow__': int_make_binop(_operator.pow),
    '__rshift__': int_make_bitop(_operator.rshift),
    '__str__': _CFT[int_str, int.__str__],
    '__sub__': int_make_binop(_operator.sub),
    '__truediv__': _CFT[int_truediv, _operator.truediv],
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

def float_abs(self_ty):
    if issubclass(self_ty, _Cst):
        return _Cst[abs(self_ty.__args__[0])]
    elif self_ty is float:
        return float
    else:
        raise TypeError

def float_bool(self_ty):
    if issubclass(self_ty, _Cst):
        return _Cst[bool(self_ty.__args__[0])]
    else:
        return bool

def float_int(self_ty):
    if issubclass(self_ty, _Cst):
        return _Cst[int(self_ty.__args__[0])]
    else:
        return int

def float_float(self_ty):
    if issubclass(self_ty, _Cst):
        return _Cst[float(self_ty.__args__[0])]
    else:
        return float

def float_str(self_ty):
    if issubclass(self_ty, _Cst):
        return _Cst[str(self_ty.__args__[0])]
    else:
        return str

def float_init(self_ty):
    if issubclass(self_ty, _Cst):
        return _Cst[float(self_ty.__args__[0])]
    else:
        from penty.penty import Types
        return Types[self_ty]['__float__'](self_ty)

def float_make_binop(operator):
    def binop(self_ty, other_ty):
        self_ty, other_ty = _astype(self_ty), _astype(other_ty)
        if self_ty is float:
            if other_ty in (bool, int, float):
                return float
            else:
                raise TypeError
        else:
            raise TypeError
        return result_types
    return _CFT[binop, operator]

def float_make_unaryop(operator):
    def unaryop(self_ty):
        if self_ty is float:
            return float
            raise TypeError
    return _CFT[unaryop, operator]


def float_make_boolop(operator):
    def boolop(self_ty, other_ty):
        self_ty, other_ty = _astype(self_ty), _astype(other_ty)
        if self_ty is float:
            if other_ty in (bool, int, float):
                return bool
            else:
                raise TypeError
        else:
            raise TypeError
    return _CFT[boolop, operator]

def float_make_biniop(operator):
    def biniop(self_ty, other_ty):
        result_ty = operator(self_ty, other_ty)
        # floatt are immutable so we don't update self_ty
        return result_ty
    return _CFT[biniop, operator]


_float_attrs = {
    '__abs__': _CFT[float_abs, float.__abs__],
    '__add__': float_make_binop(_operator.add),
    '__bool__': _CFT[float_bool, float.__bool__],
    '__eq__': float_make_boolop(_operator.eq),
    '__floordiv__': float_make_binop(_operator.floordiv),
    '__float__': _CFT[float_float, float],
    '__ge__': float_make_boolop(_operator.ge),
    '__gt__': float_make_boolop(_operator.gt),
    '__init__': _CFT[float_init, float],
    '__int__': _CFT[float_int, float.__int__],
    '__le__': float_make_boolop(_operator.le),
    '__lt__': float_make_boolop(_operator.lt),
    '__mul__': float_make_binop(_operator.mul),
    '__mod__': float_make_binop(_operator.mod),
    '__ne__': float_make_boolop(_operator.ne),
    '__neg__': float_make_unaryop(_operator.neg),
    '__pos__': float_make_unaryop(_operator.pos),
    '__pow__': float_make_binop(_operator.pow),
    '__str__': _CFT[float_str, float.__str__],
    '__sub__': float_make_binop(_operator.sub),
    '__truediv__': float_make_binop(_operator.truediv),
}

##
#
str_iterator = type(iter(""))

def str_bool(self_ty):
    if self_ty is str:
        return bool
    else:
        raise TypeError

def str_int(self_ty):
    if self_ty is str:
        return int
    else:
        raise TypeError

def str_init(self_ty):
    from penty.penty import Types
    return Types[self_ty]['__str__'](self_ty)


def str_float(self_ty):
    if self_ty is str:
        return float
    else:
        raise TypeError

def str_str(self_ty):
    if self_ty is str:
        return str
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
    '__bool__': _CFT[str_bool, bool],
    '__float__': _CFT[str_float, float],
    '__init__': _CFT[str_init, str],
    '__int__': _CFT[str_int, int],
    '__iter__': _FT[str_iter],
    '__len__': _CFT[lambda *args: int, str.__len__],
    '__str__': _CFT[str_str, str.__str__],
}

##
#

def none_eq(self_ty, other_ty):
    return _Cst[bool(other_ty is _Cst[None])]

_none_attrs = {
    '__eq__': _CFT[none_eq, _operator.eq],
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
        return _Dict[key_ty, _Cst[None]]
    else:
        return _Dict[key_ty, value_ty]

def dict_instanciate(ty):
    return {
        '__bool__': _FT[lambda *args: bool],
        '__len__': _FT[lambda *args: int],
        'clear': _FT[lambda *args: dict_clear(ty, *args)],
    }

_dict_attrs = {
    'clear': _FT[lambda x: dict_clear(x, x)],
    'from_keys': _FT[dict_fromkeys],
}

##
#

def list_append(base_ty, self_ty, value_ty):
    return _Cst[None], (_List[_astype(value_ty)], value_ty)

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
        '__bool__': _FT[lambda *args: bool],
        '__getitem__': _FT[lambda *args: list_getitem(ty, *args)],
        '__len__': _FT[lambda *args: int],
        'append': _FT[lambda *args: list_append(ty, *args)],
        'count': _FT[lambda *args: list_count(ty, *args)],
    }

_list_attrs = {
    'append': _FT[lambda self, elt: list_append(self, self, elt)],
}

##
#

def set_instanciate(ty):
    return {
        '__bool__': _FT[lambda *args: bool],
        '__len__': _FT[lambda *args: int],
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
            return _Tuple[base_value_types[key_v]]
        else:
            raise TypeError
    else:
        raise TypeError


def tuple_instanciate(ty):
    return {
        '__bool__': _FT[lambda *args: tuple_bool(ty, *args)],
        '__getitem__': _FT[lambda *args: tuple_getitem(ty, *args)],
        '__len__': _FT[lambda *args: _Cst[len(ty.__args__)]],
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
    '__next__': _FT[str_iterator_next],
}

##
#

def abs_(self_type):
    from penty.penty import Types
    return Types[self_type]['__abs__'](self_type)

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

def type_(self_ty, node=None):
    if node is None:
        return _Ty[_astype(self_ty)]
    else:
        return _TypeOf[_astype(self_ty), node.id]

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
        registry[type(None)] = _none_attrs
        registry[_FilteringBool] = _FilteringBool_attrs
        registry[str_iterator] = _str_iterator_attrs
        registry[_Dict] = dict_instanciate
        registry[_List] = list_instanciate
        registry[_Set] = set_instanciate
        registry[_Tuple] = tuple_instanciate

        registry[_Module['builtins']] = {
            'abs': {_CFT[abs_, abs]},
            'bool': {_Ty[bool]},
            'dict': {_Ty[dict]},
            'id': {_FT[id_]},
            'int': {_Ty[int]},
            'float': {_Ty[float]},
            'len': {_CFT[len_, len]},
            'repr': {_FT[repr_]},
            'slice': {_FT[slice_]},
            'type': {_FT[type_]},
        }
