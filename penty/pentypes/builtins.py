from penty.types import Cst as _Cst, Module as _Module, Type as _Ty
from penty.types import astype as _astype, TypeOf as _TypeOf, asset as _asset
from penty.types import ConstFunctionType as _CFT, Tuple as _Tuple
from penty.types import FunctionType as _FT, Generator as _Generator
from penty.types import MethodType as _MT, FDef as _FDef, Lambda as _Lambda
from penty.types import FilteringBool as _FilteringBool, get_typer as _get_typer
from penty.types import List as _List, Set as _Set, Dict as _Dict
from penty.types import SetIterator as _SetIterator
from penty.types import DictKeyIterator as _DictKeyIterator
from penty.types import DictValueIterator as _DictValueIterator
from penty.types import DictItemIterator as _DictItemIterator
from penty.types import ListIterator as _ListIterator
from penty.types import resolve_base_attrs
import itertools as _itertools
import operator as _operator

import sys

##
#

def int_divmod(self, value):
    from penty.penty import Types
    fd = Types[_Module['operator']]['__floordiv__'](self, value)
    m = Types[_Module['operator']]['__mod__'](self, value)
    return _Tuple[fd, m]

def int_init(value=None, base=None):
    if value is None:
        return _Cst[0]

    if base is not None:
        if not issubclass(_astype(base), int):
            raise TypeError

    from penty.penty import Types
    if issubclass(value, (int, float, str)):
        return int

    if '__int__' in Types[value]:
        return Types[value]['__int__'](_astype(value))
    raise TypeError

def int_make_binop(operator):
    def binop(self, value):
        self, value = _astype(self), _astype(value)
        if issubclass(self, int) and issubclass(value, int):
            return int
        else:
            raise TypeError
    return _CFT[binop, operator]

def int_make_bitop(operator):
    def binop(self, value):
        self, value = _astype(self), _astype(value)
        if issubclass(self, int) and issubclass(value, int):
            return int
        else:
            raise TypeError
    return _CFT[binop, operator]

def int_truediv(self, value):
    self, value = _astype(self), _astype(value)
    if issubclass(self, int) and issubclass(value, int):
        return float
    else:
        raise TypeError

def int_make_unaryop(operator):
    def unaryop(self):
        if issubclass(self, int):
            return int
        else:
            raise TypeError
    return _CFT[unaryop, operator]

def int_bit_length(self):
    if issubclass(self, int):
        return int
    else:
        raise TypeError

def int_conjugate(self):
    if issubclass(self, int):
        return int
    else:
        raise TypeError

def int_bool(self):
    if issubclass(self, int):
        return bool
    else:
        raise TypeError

def int_eq(self, value):
    if issubclass(_astype(self), int):
        return bool
    else:
        raise TypeError

def int_ceil(self):
    if issubclass(self, int):
        return int
    else:
        raise TypeError

def int_int(self):
    if issubclass(self, int):
        return int
    else:
        raise TypeError

def int_abs(self):
    if issubclass(self, int):
        return int
    else:
        raise TypeError

def int_float(self):
    if issubclass(self, int):
        return float
    else:
        raise TypeError

def int_hash(self):
    if issubclass(self, int):
        return int
    else:
        raise TypeError

def int_ne(self, value):
    if issubclass(_astype(self), int):
        return bool
    else:
        raise TypeError


def int_str(self):
    if issubclass(self, int):
        return str
    else:
        raise TypeError

def int_make_boolop(operator):
    def boolop(self, value):
        self, value = _astype(self), _astype(value)
        if issubclass(self, int) and issubclass(value, (int, float)):
            return bool
        else:
            raise TypeError
    return _CFT[boolop, operator]

def int_make_biniop(operator):
    def biniop(self, value):
        result_ty = operator(self, value)
        # int are immutable so we don't update self
        return result_ty
    return _CFT[biniop, operator]

def int_pow(self, value, mod=_Cst[None]):
    self, value, mod = _astype(self), _astype(value), _astype(mod)
    if not issubclass(self, int):
        raise TypeError
    if mod is not _Cst[None]:
        if not issubclass(value, int):
            raise TypeError
        if not issubclass(mod, int):
            raise TypeError
        return int
    if issubclass(value, int):
        return int
    elif issubclass(value, float):
        return value
    else:
        raise TypeError


_int_attrs = {
    '__abs__': _CFT[int_abs, int.__abs__],
    '__add__': int_make_binop(_operator.add),
    '__and__': int_make_bitop(_operator.and_),
    '__bases__': _Tuple[_Ty[object]],
    '__bool__': _CFT[int_bool, int.__bool__],
    '__ceil__': _CFT[int_ceil, int.__ceil__],
    '__divmod__': _CFT[int_divmod, int.__divmod__],
    '__eq__': _CFT[int_eq, int.__eq__],
    '__float__': _CFT[int_float, int.__float__],
    '__floordiv__': int_make_binop(_operator.floordiv),
    '__ge__': int_make_boolop(_operator.ge),
    '__gt__': int_make_boolop(_operator.gt),
    '__hash__': _CFT[int_hash, int.__hash__],
    '__int__': _CFT[int_int, int],
    '__init__': _CFT[int_init, int],
    '__invert__': int_make_unaryop(_operator.inv),
    '__le__': int_make_boolop(_operator.le),
    '__lshift__': int_make_bitop(_operator.lshift),
    '__lt__': int_make_boolop(_operator.lt),
    '__mul__': int_make_binop(_operator.mul),
    '__mod__': int_make_binop(_operator.mod),
    '__name__': _Cst['int'],
    '__ne__': _CFT[int_ne, int.__ne__],
    '__neg__': int_make_unaryop(_operator.neg),
    '__or__': int_make_bitop(_operator.or_),
    '__pos__': int_make_unaryop(_operator.pos),
    '__pow__': _CFT[int_pow, int.__pow__],
    '__rshift__': int_make_bitop(_operator.rshift),
    '__str__': _CFT[int_str, int.__str__],
    '__sub__': int_make_binop(_operator.sub),
    '__truediv__': _CFT[int_truediv, _operator.truediv],
    '__xor__': int_make_bitop(_operator.xor),
    'bit_length': _CFT[int_bit_length, int.bit_length],
    'conjugate': _CFT[int_conjugate, int.conjugate],
    'denominator': _Cst[1],
    'imag': _Cst[0],
    'numerator': int,
    'real': int,
}

##
#


def bool_and(self, value):
    self, value = _astype(self), _astype(value)
    if self is bool and value is bool:
        return bool
    if all(issubclass(ty, (bool, int)) for ty in (self, value)):
        return int
    raise TypeError

def bool_init(value=_Cst[False]):
    from penty.penty import Types
    if _astype(value) is bool:
        return value
    type_ = Types[_astype(value)]
    if '__bool__' in type_:
        return type_['__bool__'](value)
    if '__len__' in type_:
        len_ty = type_['__len__'](value)
        if issubclass(len_ty, _Cst):
            return _Cst[bool(len_ty.__args__[0])]
        return bool
    raise TypeError

def bool_not(self):
    if self is bool:
        return bool
    else:
        raise TypeError

def bool_or(self, value):
    return bool_and(self, value)

def bool_str(self):
    return str

def bool_xor(self, value):
    return bool_and(self, value)


_bool_attrs = {
    '__and__': _CFT[bool_and, bool.__and__],
    '__bases__': _Tuple[_Ty[int]],
    '__init__': _CFT[bool_init, bool],
    '__name__': _Cst['bool'],
    '__or__': _CFT[bool_or, bool.__or__],
    '__str__': _CFT[bool_str, bool.__str__],
    '__xor__': _CFT[bool_xor, bool.__xor__],
}


##
#

def FilteringBool_bool(self):
    return _Cst[self.__args__[0]]

def FilteringBool_not(self):
    val, id_, filtered_tys = self.__args__
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

def float_abs(self):
    return float

def float_bool(self):
    return bool

def float_divmod(self, value):
    return _Tuple[_float_attrs['__floordiv__'](self, value),
                  _float_attrs['__mod__'](self, value)]

def float_float(self):
    return float

def float_hash(self):
    if not issubclass(self, float):
        raise TypeError
    return int

def float_init(value):
    from penty.penty import Types
    value = _astype(value)
    if issubclass(value, (bool, int, float, str)):
        return float
    if '__float__' in Types[value]:
        return Types[value]['__float__'](value)
    raise TypeError

def float_int(self):
    return int

def float_make_binop(operator):
    def binop(self, value):
        self, value = _astype(self), _astype(value)
        if self is float and value in (bool, int, float):
            return float
        else:
            raise TypeError
    return _CFT[binop, operator]

def float_make_unaryop(operator):
    def unaryop(self):
        if self is float:
            return float
        raise TypeError
    return _CFT[unaryop, operator]


def float_make_boolop(operator):
    def boolop(self, value):
        self, value = _astype(self), _astype(value)
        if self is float and value in (bool, int, float):
            return bool
        else:
            raise TypeError
    return _CFT[boolop, operator]

def float_pow(self, value, mod=_Cst[None]):
    self, value = _astype(self), _astype(value)
    if mod is not _Cst[None]:
        raise TypeError
    if issubclass(self, float) and issubclass(value, (int, float)):
        return float
    else:
        raise TypeError


def float_str(self):
    return str


_float_attrs = {
    '__abs__': _CFT[float_abs, float.__abs__],
    '__add__': float_make_binop(_operator.add),
    '__bases__': _Tuple[_Ty[object]],
    '__bool__': _CFT[float_bool, float.__bool__],
    '__divmod__': _CFT[float_divmod, float.__divmod__],
    '__eq__': float_make_boolop(_operator.eq),
    '__float__': _CFT[float_float, float],
    '__floordiv__': float_make_binop(_operator.floordiv),
    '__ge__': float_make_boolop(_operator.ge),
    '__gt__': float_make_boolop(_operator.gt),
    '__hash__': _CFT[float_hash, float.__hash__],
    '__init__': _CFT[float_init, float],
    '__int__': _CFT[float_int, float.__int__],
    '__le__': float_make_boolop(_operator.le),
    '__lt__': float_make_boolop(_operator.lt),
    '__mul__': float_make_binop(_operator.mul),
    '__mod__': float_make_binop(_operator.mod),
    '__name__': _Cst['float'],
    '__ne__': float_make_boolop(_operator.ne),
    '__neg__': float_make_unaryop(_operator.neg),
    '__pos__': float_make_unaryop(_operator.pos),
    '__pow__': _CFT[float_pow, float.__pow__],
    '__str__': _CFT[float_str, float.__str__],
    '__sub__': float_make_binop(_operator.sub),
    '__truediv__': float_make_binop(_operator.truediv),
}

for slot in ('add', 'divmod', 'floordiv', 'mod', 'mul', 'pow', 'sub', 'truediv'):
    _float_attrs['__r{}__'.format(slot)] = _float_attrs['__{}__'.format(slot)]

##
#

def complex_abs(self):
    if not issubclass(self, complex):
        raise TypeError
    return complex

def complex_bool(self):
    if not issubclass(self, complex):
        raise TypeError
    return bool

def complex_divmod(self, value):
    if not issubclass(self, complex):
        raise TypeError
    raise TypeError

def complex_hash(self):
    if not issubclass(self, complex):
        raise TypeError
    return int

def complex_init(real_ty, imag_ty=None):
    from penty.penty import Types
    real_ty = _astype(real_ty)

    if imag_ty is None:
        if issubclass(real_ty, complex):
            return complex
        Types[float]['__init__'](real_ty)
        return complex

    # interestingly, if the real part is a complex, everything is fine (!)
    imag_ty = _astype(imag_ty)
    if issubclass(real_ty, complex):
        if issubclass(imag_ty, complex):
            return complex
        Types[imag_ty]['__float__'](imag_ty)
        return complex

    Types[real_ty]['__float__'](real_ty)
    if issubclass(imag_ty, complex):
        return complex
    Types[imag_ty]['__float__'](imag_ty)
    return complex

def complex_eq(self, value):
    if not issubclass(self, complex):
        raise TypeError
    return bool

def complex_ne(self, value):
    if not issubclass(self, complex):
        raise TypeError
    return bool

def complex_int(self):
    if not issubclass(self, complex):
        raise TypeError
    raise TypeError

def complex_make_binop(operator):
    def binop(self, value):
        self, value = _astype(self), _astype(value)
        if not issubclass(self, complex):
            raise TypeError

        if not issubclass(value, (int, float, complex)):
            raise TypeError

        return complex

    return _CFT[binop, operator]

def complex_make_unaryop(operator):
    def unaryop(self):
        if not issubclass(self, complex):
            raise TypeError
        return complex
    return _CFT[unaryop, operator]


def complex_make_boolop(operator):
    def boolop(self, value):
        if not issubclass(self, complex):
            raise TypeError
        raise TypeError
    return _CFT[boolop, operator]

def complex_pow(self, value, mod=_Cst[None]):
    self, value = _astype(self), _astype(value)
    if mod is not _Cst[None]:
        raise TypeError
    if issubclass(self, complex) and issubclass(value, (int, float, complex)):
        return complex
    else:
        raise TypeError


def complex_str(self):
    if not issubclass(self, complex):
        raise TypeError
    return str

_complex_attrs = {
    '__abs__': _CFT[complex_abs, complex.__abs__],
    '__add__': complex_make_binop(_operator.add),
    '__bases__': _Tuple[_Ty[object]],
    '__bool__': _CFT[complex_bool, complex.__bool__],
    '__divmod__': _FT[complex_divmod],
    '__eq__': _CFT[complex_eq, _operator.eq],
    '__floordiv__': _FT[complex_divmod],  # same as floordiv: type error
    '__ge__': complex_make_boolop(_operator.ge),
    '__gt__': complex_make_boolop(_operator.gt),
    '__hash__': _CFT[complex_hash, complex.__hash__],
    '__init__': _CFT[complex_init, complex],
    '__int__': _CFT[complex_int, complex.__int__],
    '__le__': complex_make_boolop(_operator.le),
    '__lt__': complex_make_boolop(_operator.lt),
    '__mul__': complex_make_binop(_operator.mul),
    '__mod__': _FT[complex_divmod],  # same as floordiv: type error
    '__name__': _Cst['complex'],
    '__ne__': _CFT[complex_ne, _operator.ne],
    '__neg__': complex_make_unaryop(_operator.neg),
    '__pos__': complex_make_unaryop(_operator.pos),
    '__pow__': _CFT[complex_pow, complex.__pow__],
    '__str__': _CFT[complex_str, complex.__str__],
    '__sub__': complex_make_binop(_operator.sub),
    '__truediv__': complex_make_binop(_operator.truediv),
}

for slot in ('add', 'divmod', 'floordiv', 'mod', 'mul', 'pow', 'sub', 'truediv'):
    _complex_attrs['__r{}__'.format(slot)] = _complex_attrs['__{}__'.format(slot)]

##
#

range_iterator = type(iter(range(1)))

def range_bool(self):
    return bool

def range_contains(self, key):
    return bool

def range_count(self, value):
    return int

def range_getitem(self, key):
    from penty.penty import Types
    key_ty = _astype(key)
    if issubclass(key_ty, int):
        return int
    elif issubclass(key_ty, slice):
        return self
    elif '__index__' in Types[key_ty]:
        # PEP-0357 indexing
        if issubclass(Types[key_ty]['__index__'](key_ty), int):
            return int
    raise TypeError

def range_hash(self):
    return int

def range_init(bound0, bound1=None, step=None):
    from penty.penty import Types
    for p in (bound0, bound1, step):
        if p is None:
            continue
        p = _astype(p)
        if '__index__' in Types[p]:
            p = Types[p]['__index__'](p)
        if issubclass(p, int):
            continue
        raise TypeError
    return range

def range_iter(self):
    return range_iterator

def range_len(self):
    return int

def range_str(self):
    return str

def range_index(self, value, start=_Cst[0], stop=_Cst[sys.maxsize]):
    if not issubclass(_astype(start), int):
        raise TypeError
    if not issubclass(_astype(stop), int):
        raise TypeError
    return int

def make_range_compare():
    def range_compare(self, value):
        if not issubclass(value, range):
            raise TypeError
        return bool
    return _MT[range, range_compare]

def range_ne(self, value):
    if not issubclass(value, range):
        return _Cst[True]
    return bool

def range_eq(self, value):
    if not issubclass(value, range):
        return _Cst[False]
    return bool

def range_reversed(self):
    return range_iterator

_range_attrs = {
    '__bases__': _Tuple[_Ty[object]],
    '__bool__': _MT[range, range_bool],
    '__contains__': _FT[range_contains],
    '__eq__': _MT[range, range_eq],
    '__ge__': make_range_compare(),
    '__getitem__': _MT[range, range_getitem],
    '__gt__': make_range_compare(),
    '__hash__': _MT[range, range_hash],
    '__init__': _FT[range_init],
    '__iter__': _MT[range, range_iter],
    '__le__': make_range_compare(),
    '__len__': _MT[range, range_len],
    '__lt__': make_range_compare(),
    '__name__': _Cst['range'],
    '__ne__': _MT[range, range_ne],
    '__reversed__': _MT[range, range_reversed],
    '__str__': _MT[range, range_str],
    'count': _MT[range, range_count],
    'index': _MT[range, range_index],
    'start': int,
    'step': int,
    'stop': int,
}


##
#
str_iterator = type(iter(""))

def str_add(self, value):
    value = _astype(value)
    if issubclass(self, str) and issubclass(value, str):
        return str
    raise TypeError

def str_bool(self):
    if self is str:
        return bool
    else:
        raise TypeError

def str_hash(self):
    if self is str:
        return int
    else:
        raise TypeError

def str_int(self):
    if self is str:
        return int
    else:
        raise TypeError

def str_init(self):
    from penty.penty import Types
    return Types[self]['__str__'](self)


def str_float(self):
    if self is str:
        return float
    else:
        raise TypeError

def str_str(self):
    if self is str:
        return str
    else:
        raise TypeError


def str_iter(self):
    if _astype(self) is str:
        return str_iterator
    else:
        raise TypeError

def str_len(self):
    if self is str:
        return int
    else:
        raise TypeError

def str_lt(self, value):
    if _astype(self) is str and _astype(value) is str:
        return bool
    else:
        raise TypeError

_str_attrs = {
    '__add__': _CFT[str_add, str.__add__],
    '__bases__': _Tuple[_Ty[object]],
    '__hash__': _CFT[str_hash, str.__hash__],
    '__init__': _CFT[str_init, str],
    '__iter__': _FT[str_iter],
    '__len__': _CFT[str_len, str.__len__],
    '__lt__': _CFT[str_lt, str.__lt__],
    '__name__': _Cst['str'],
    '__str__': _CFT[str_str, str.__str__],
}

##
#

def none_eq(self, value):
    return _Cst[bool(value is _Cst[None])]

_none_attrs = {
    '__eq__': _CFT[none_eq, _operator.eq],
    '__name__': _Cst['NoneType'],
    '__bases__': _Tuple[_Ty[object]],
}

##
#

def dict_contains(self, key):
    if not self.__args__[0]:
        return _Cst[False]
    return bool

def dict_clear(self):
    return _Cst[None]

def dict_copy(self):
    return _Dict[self.__args__[0].copy(),
                 self.__args__[1].copy()]

def dict_delitem(self, key):
    if not self.__args__[0]:
        raise TypeError
    return _Cst[None]

def dict_get(self, key, default=_Cst[None]):
    return self.__args__[1].union([default])

def dict_getitem(self, key):
    return self.__args__[1].copy()

def dict_pop(self, key, default=_Cst[None]):
    return dict_get(self, key, default)

def dict_popitem(self):
    return {_Tuple[key, value_ty]
            for key, value_ty in _itertools.product(*self.__args__)}

def dict_items(self):
    return _DictItemIterator[{_Tuple[kty, vty]
                              for kty, vty in
                              _itertools.product(*self.__args__)}]

def dict_keys(self):
    return _DictKeyIterator[self.__args__[0]]

def dict_len(self):
    if not self.__args__[0]:
        return _Cst[0]
    return int

def dict_setdefault(self, key, default=_Cst[None]):
    default = _astype(default)

    self.__args__[0].add(_astype(key))
    self.__args__[1].add(default)

    return dict_get(self, key, default)

def dict_setitem(self, key, value):
    self.__args__[0].add(key)
    self.__args__[1].add(value)
    return _Cst[None]

def dict_update(self, *other_tys):
    from penty.penty import Types

    for other_ty in other_tys:
        other_ty = _astype(other_ty)
        if issubclass(other_ty, _Dict):
            self.__args__[0].update(other_ty.__args__[0])
            self.__args__[1].update(other_ty.__args__[1])
            continue

        iter_tys = iter_(other_ty)
        if not isinstance(iter_tys, set):
            iter_tys = {iter_tys}
        for iter_ty in iter_tys:
            value_tys = next_(iter_ty)
            if not isinstance(value_tys, set):
                value_tys = {value_tys}
            for value_ty in value_tys:
                if not issubclass(value_ty, _Tuple):
                    raise TypeError
                if len(value_ty.__args__) != 2:
                    raise TypeError
                hash_(_astype(value_ty.__args__[0]))
                self.__args__[0].add(_astype(value_ty.__args__[0]))
                self.__args__[1].add(_astype(value_ty.__args__[1]))
    return _Cst[None]

def dict_eq(self, value):
    if not issubclass(value, dict):
        return _Cst[False]
    if bool(self.__args__[0]) ^ bool(value.__args__[0]):
        return _Cst[False]
    return bool

def dict_init(items_ty=None):
    from penty.penty import Types
    result_ty = _Dict[set(), set()]
    if items_ty is not None:
        dict_update(result_ty, items_ty)
    return result_ty

def dict_iter(self):
    return _DictKeyIterator[self.__args__[0]]

def dict_fromkeys(iterable, value=_Cst[None]):
    from penty.penty import Types

    iter_ty = iter_(iterable)
    key = next_(iter_ty)

    return _Dict[key, value]

def make_dict_compare():
    def dict_compare(self, value):
        return bool
    return _MT[dict, dict_compare]

def dict_ne(self, value):
    if not issubclass(value, dict):
        return _Cst[True]
    if bool(self.__args__[0]) ^ bool(value.__args__[0]):
        return _Cst[True]
    return bool

def dict_values(self):
    return _DictValueIterator[self.__args__[1]]

def dict_instanciate(ty):
    keys, values = ty.__args__
    for key in keys:
        hash_(key)
    return _dict_methods

_dict_methods = {
    '__contains__': _MT[dict, dict_contains],
    '__delitem__': _MT[dict, dict_delitem],
    '__getitem__': _MT[dict, dict_getitem],
    '__hash__': _Cst[None],
    '__init__': _FT[dict_init],
    '__iter__': _MT[dict, dict_iter],
    '__len__': _MT[dict, dict_len],
    '__gt__': make_dict_compare(),
    '__ge__': make_dict_compare(),
    '__lt__': make_dict_compare(),
    '__le__': make_dict_compare(),
    '__eq__': _MT[dict, dict_eq],
    '__ne__': _MT[dict, dict_ne],
    '__setitem__': _MT[dict, dict_setitem],
    'clear': _MT[dict, dict_clear],
    'copy': _MT[dict, dict_copy],
    'get': _MT[dict, dict_get],
    'items': _MT[dict, dict_items],
    'keys': _MT[dict, dict_keys],
    'pop': _MT[dict, dict_pop],
    'popitem': _MT[dict, dict_popitem],
    'setdefault': _MT[dict, dict_setdefault],
    'update': _MT[dict, dict_update],
    'values': _MT[dict, dict_values],
}

_dict_attrs = {
    '__bases__': _Tuple[_Ty[object]],
    '__name__': _Cst['dict'],
    'fromkeys': _FT[dict_fromkeys],
}
_dict_attrs.update(_dict_methods)

##
#

def dict_item_iterator_iter(self):
    # not exactly correct, python use a temporary type here
    return self

def dict_item_iterator_next(self):
    return set(self.__args__[0])

def dict_item_iterator_instanciate(ty):
    return {
        '__iter__': _FT[dict_item_iterator_iter],
        '__next__': _FT[dict_item_iterator_next],
    }

##
#

def dict_key_iterator_iter(self):
    # not exactly correct, python use a temporary type here
    return self

def dict_key_iterator_next(self):
    return set(self.__args__[0])

def dict_key_iterator_instanciate(ty):
    return {
        '__iter__': _FT[dict_key_iterator_iter],
        '__next__': _FT[dict_key_iterator_next],
    }
##
#

def dict_value_iterator_iter(self):
    # not exactly correct, python use a temporary type here
    return self

def dict_value_iterator_next(self):
    return set(self.__args__[0])

def dict_value_iterator_instanciate(ty):
    return {
        '__iter__': _FT[dict_value_iterator_iter],
        '__next__': _FT[dict_value_iterator_next],
    }

##
#

def list_add(self, value):
    if not issubclass(value, list):
        raise TypeError
    return _List[self.__args__[0].union(value.__args__[0])]

def list_append(self, object):
    self.__args__[0].add(_astype(object))
    return _Cst[None]

def list_contains(self, key):
    # We cannot compare self.__args__[0] with key.__args__[0] for non
    # empty intersection since values can evaluate equal while having different
    # types (1 == 1.0)
    return bool

def list_delitem(self, key):
    from penty.penty import Types
    idx_ty = _astype(key)
    if issubclass(idx_ty, int):
        return _Cst[None]
    elif issubclass(idx_ty, int):
        return _Cst[None]
    elif '__index__' in Types[idx_ty]:
        # PEP-0357 indexing
        if issubclass(Types[idx_ty]['__index__'](idx_ty), int):
            return Cst[None]
    raise TypeError

def list_eq(self, value):
    if not issubclass(value, list):
        return _Cst[False]
    # Since `[1] == [1.0]`, we cannot inspect types of elements of self and
    # other
    return bool

def list_bool_binop(self, value):
    if not issubclass(value, list):
        raise TypeError
    return bool

def list_getitem(self, idx):
    from penty.penty import Types
    idx_ty = _astype(idx)
    if issubclass(idx_ty, int):
        return self.__args__[0]
    elif issubclass(idx_ty, slice):
        return self
    elif '__index__' in Types[idx_ty]:
        # PEP-0357 indexing
        if issubclass(Types[idx_ty]['__index__'](idx_ty), int):
            return self.__args__[0]
    raise TypeError

def list_iadd(self, value):
    list_extend(self, value)
    return self

def list_imul(self, value):
    if not issubclass(_astype(value), int):
        raise TypeError
    return self

def list_init(elts_ty=None):
    from penty.penty import Types
    if elts_ty is None:
        return _List[set()]
    iter_ty = iter_(elts_ty)
    next_ty = next_(iter_ty)
    return _List[next_ty]

def list_iter(self):
    return _ListIterator[self.__args__[0]]

def list_len(self):
    if not self.__args__[0]:
        return _Cst[0]
    return int

def list_mul(self, value):
    if not issubclass(_astype(value), int):
        raise TypeError
    return self

def list_ne(self, value):
    if not issubclass(value, list):
        return _Cst[True]
    return bool

def list_reversed(self):
    return _ListIterator[self.__args__[0]]

def list_setitem(self, key, value):
    if issubclass(_astype(key), int):
        self.__args__[0].add(_astype(value))
    elif _astype(key) is slice:
        # value should be iterable
        from penty.penty import Types
        iter_ty = iter_(value)
        next_ty = next_(iter_ty)
        self.__args__[0].update(next_ty)
    return _Cst[None]

def list_str(self):
    return str

def list_clear(self):
    return _Cst[None]

def list_copy(self):
    return _List[self.__args__[0].copy()]

def list_count(self, value):
    if self.__args__[0] == set():
        return _Cst[0]
    return int

def list_extend(self, iterable):
    from penty.penty import Types
    iter_ty = iter_(iterable)
    next_ty = next_(iter_ty)
    self.__args__[0].update(next_ty)
    return _Cst[None]

def list_index(self, value, start=_Cst[0], stop=_Cst[sys.maxsize]):
    if not issubclass(_astype(start), int):
        raise TypeError
    if not issubclass(_astype(stop), int):
        raise TypeError
    return int

def list_insert(self, index, object):
    if not issubclass(_astype(index), int):
        raise TypeError
    self.__args__[0].add(object)
    return _Cst[None]

def list_pop(self, index=_Cst[-1]):
    if not issubclass(_astype(index), int):
        raise TypeError
    return self.__args__[0].copy()

def list_remove(self, value):
    #a = [1.0] ; a.remove(1) ; a == []
    return _Cst[None]

def list_reverse(self):
    return _Cst[None]

def list_sort(self, *, key=_Cst[None], reverse=_Cst[False]):
    from penty.penty import Types
    typer_instance = _get_typer()

    # Find possible return values of the key fct
    if key is not _Cst[None]:
        key_ty = set()
        for elt_ty in self.__args__[0]:
            key_ty.update(_asset(typer_instance._call(key, elt_ty)))

        # check that they are all comparable
        for left, right in _itertools.product(key_ty, key_ty):
            Types[_Module['operator']]['lt'](left, right)

    if not issubclass(_astype(reverse), int):
        raise TypeError

    return _Cst[None]

def list_instanciate(ty):
    return _list_methods

_list_methods = {
    '__add__': _MT[list, list_add],
    '__contains__': _MT[list, list_contains],
    '__delitem__': _MT[list, list_delitem],
    '__eq__': _MT[list, list_eq],
    '__ge__': _MT[list, list_bool_binop],
    '__getitem__': _MT[list, list_getitem],
    '__gt__': _MT[list, list_bool_binop],
    '__hash__': _Cst[None],
    '__iadd__': _MT[list, list_iadd],
    '__imul__': _MT[list, list_imul],
    '__iter__': _MT[list, list_iter],
    '__le__': _MT[list, list_bool_binop],
    '__len__': _MT[list, list_len],
    '__lt__': _MT[list, list_bool_binop],
    '__mul__': _MT[list, list_mul],
    '__ne__': _MT[list, list_ne],
    '__reversed__': _MT[list, list_reversed],
    '__rmul__': _MT[list, list_mul],
    '__setitem__': _MT[list, list_setitem],
    '__str__': _MT[list, list_str],
    'append': _MT[list, list_append],
    'clear': _MT[list, list_clear],
    'copy': _MT[list, list_copy],
    'count': _MT[list, list_count],
    'extend': _MT[list, list_extend],
    'index': _MT[list, list_index],
    'insert': _MT[list, list_insert],
    'pop': _MT[list, list_pop],
    'remove': _MT[list, list_remove],
    'reverse': _MT[list, list_reverse],
    'sort': _MT[list, list_sort],
}

_list_attrs = _list_methods.copy()
_list_attrs.update({
    '__init__': _FT[list_init],
    '__bases__': _Tuple[_Ty[object]],
    '__name__': _Cst['list'],
})


def list_iterator_next(self_ty):
    return set(self_ty.__args__[0])

def list_iterator_instanciate(ty):
    return {
        '__next__': _FT[list_iterator_next],
    }

##
#

_object_attrs = {
}

##
#

def set_and(self, value):
    if not self.__args__[0]:
        return _Set[{}]
    return _Set[self.__args__[0] | value.__args__[0]]

def make_set_compare():
    def set_compare(self, value):
        return bool
    return _MT[set, set_compare]

def set_iand(self, value):
    if not issubclass(value, set):
        raise TypeError
    self.__args__[0] |= value.__args__[0]
    return self

def set_init(elts_ty=None):
    from penty.penty import Types
    if elts_ty is None:
        return _Set[set()]
    iter_ty = iter_(elts_ty)
    next_ty = _asset(next_(iter_ty))
    for ty in next_ty:
        hash_(ty)
    return _Set[next_ty]

def set_ior(self, value):
    if not issubclass(value, set):
        raise TypeError
    self.__args__[0] |= value.__args__[0]
    return self

def set_isub(self, value):
    if not issubclass(value, set):
        raise TypeError
    return self

def set_ixor(self, value):
    if not issubclass(value, set):
        raise TypeError
    self.__args__[0] |= value.__args__[0]
    return self

def set_len(self):
    if not self.__args__[0]:
        return _Cst[0]
    return int

def set_iter(self):
    return _SetIterator[self.__args__[0]]

def set_or(self, value):
    if not issubclass(value, set):
        raise TypeError
    return _Set[self.__args__[0] | value.__args__[0]]

def set_sub(self, value):
    if not issubclass(value, set):
        raise TypeError
    return _Set[self.__args__[0]]

def set_xor(self, value):
    if not issubclass(value, set):
        raise TypeError
    return _Set[self.__args__[0] | value.__args__[0]]

def set_add(self, value_ty):
    self.__args__[0].add(_astype(value_ty))
    return _Cst[None]

def set_bool(self):
    if not self.__args__[0]:
        return _Cst[False]
    return bool

def set_clear(self):
    return _Cst[None]

def set_contains(self, value_ty):
    if not self.__args__[0]:
        return _Cst[False]
    return bool

def set_copy(self):
    return _Set[self.__args__[0].copy()]

def set_difference(self, *values):
    from penty.penty import Types
    for value in values:
        iter_(value)
    return _Set[self.__args__[0].copy()]

def set_difference_update(self, *values):
    set_difference(self, *values)
    return _Cst[None]

def set_discard(self, elt_ty):
    if not self.__args__[0]:
        raise TypeError
    return _Cst[None]

def set_eq(self, value):
    if not issubclass(value, set):
        return _Cst[False]
    if bool(self.__args__[0]) ^ bool(value.__args__[0]):
        return _Cst[False]
    return bool

def set_intersection(self, *values):
    from penty.penty import Types
    intersection_tys = self.__args__[0].copy()
    for value in values:
        iter_tys = iter_(value)
        if not isinstance(iter_tys, set):
            iter_tys = {iter_tys}
        for iter_ty in iter_tys:
            value_tys = next_(iter_ty)
            if not isinstance(value_tys, set):
                value_tys = {value_tys}
            # still taking the union and not the intersection, because of cases
            # like {1} & {1.}
            intersection_tys.update(value_tys)
    return _Set[intersection_tys]

def set_intersection_update(self, *values):
    updated = set_intersection(self, *values)
    self.__args__[0].update(updated.__args__[0])
    return _Cst[None]

def set_isdisjoint(self, value):
    from penty.penty import Types
    if not self.__args__[0] or not value.__args__[0]:
        return _Cst[True]
    iter_(value)
    return bool

def set_issubset(self, value):
    from penty.penty import Types
    if not self.__args__[0]:
        return _Cst[True]
    iter_(value)
    return bool

def set_issuperset(self, value):
    if not self.__args__[0] and not value.__args__[0]:
        return _Cst[True]
    return set_issubset(self, value)

def set_ne(self, value):
    if not issubclass(value, set):
        return _Cst[True]
    if bool(self.__args__[0]) ^ bool(value.__args__[0]):
        return _Cst[True]
    return bool

def set_pop(self):
    if not self.__args__[0]:
        raise TypeError
    return self.__args__[0].copy()

def set_remove(self, value_ty):
    if not self.__args__[0]:
        raise TypeError
    return _Cst[None]

def set_symmetric_difference(self, value):
    return set_intersection(self, value)

def set_symmetric_difference_update(self, value):
    return set_intersection_update(self, value)

def set_union(self, *values):
    return set_intersection(self, *values)

def set_update(self, *values):
    return set_intersection_update(self, *values)

def set_instanciate(ty):
    keys, = ty.__args__
    for key in keys:
        hash_(key)
    return _set_methods

_set_methods = {
    '__and__': _MT[set, set_and],
    '__contains__': _MT[set, set_contains],
    '__eq__': _MT[set, set_eq],
    '__ge__': make_set_compare(),
    '__gt__': make_set_compare(),
    '__hash__': _Cst[None],
    '__iand__': _MT[set, set_iand],
    '__ior__': _MT[set, set_ior],
    '__isub__': _MT[set, set_isub],
    '__iter__': _MT[set, set_iter],
    '__ixor__': _MT[set, set_ixor],
    '__le__': make_set_compare(),
    '__len__': _MT[set, set_len],
    '__lt__': make_set_compare(),
    '__ne__': _MT[set, set_ne],
    '__or__': _MT[set, set_or],
    '__rand__': _MT[set, set_and],
    '__ror__': _MT[set, set_or],
    '__rsub__': _MT[set, set_sub],
    '__rxor__': _MT[set, set_xor],
    '__sub__': _MT[set, set_sub],
    '__xor__': _MT[set, set_xor],
    'add': _MT[set, set_add],
    'clear': _MT[set, set_clear],
    'copy': _MT[set, set_copy],
    'difference': _MT[set, set_difference],
    'difference_update': _MT[set, set_difference_update],
    'discard': _MT[set, set_discard],
    'intersection': _MT[set, set_intersection],
    'intersection_update': _MT[set, set_intersection_update],
    'isdisjoint': _MT[set, set_isdisjoint],
    'issubset': _MT[set, set_issubset],
    'issuperset': _MT[set, set_issuperset],
    'pop': _MT[set, set_pop],
    'remove': _MT[set, set_remove],
    'symmetric_difference': _MT[set, set_symmetric_difference],
    'symmetric_difference_update': _MT[set, set_symmetric_difference_update],
    'union': _MT[set, set_union],
    'update': _MT[set, set_update],
}

_set_attrs = _set_methods.copy()
_set_attrs.update({
    '__bases__': _Tuple[_Ty[object]],
    '__init__': _FT[set_init],
    '__name__': _Cst['set'],
})

##
#

def set_iterator_next(self):
    return set(self.__args__[0])

def set_iterator_instanciate(ty):
    return {
        '__next__': _FT[set_iterator_next],
    }

##
#

def tuple_getitem(self, key):
    base_value_types = self.__args__

    if key in (bool, int):
        return set(base_value_types)
    elif key is slice:
        return tuple
    elif issubclass(key, _Cst):
        key_v = key.__args__[0]
        if isinstance(key_v, (bool, int)):
            return base_value_types[key_v]
        elif isinstance(key_v, slice):
            return _Tuple[base_value_types[key_v]]
        else:
            raise TypeError
    else:
        raise TypeError

def tuple_init(iterable=None):
    from penty.penty import Types

    if iterable is None:
        return _Tuple[()]
    if issubclass(iterable, tuple):
        return iterable
    if issubclass(iterable, _Cst):
        ctuple = tuple(iterable.__args__[0])
        return _Tuple[tuple(type(x) for x in ctuple)]
    iterable_ty = Types[iterable]
    if '__len__' in iterable_ty and iterable_ty['__len__'](iterable) is _Cst[0]:
        return _Tuple[()]
    iter_ty = iter_(iterable)
    elts_ty = _asset(next_(iter_ty))
    # FIXME: should we have frozen sets as tuple elts?
    return {_Tuple[elt_ty, ...] for elt_ty in elts_ty}

def tuple_hash(self):
    for ty in self.__args__:
        hash_(_astype(ty))
    return int


def tuple_len(self):
    if self.__args__[-1] is Ellipsis:
        return int
    return _Cst[len(self.__args__)]

_tuple_methods = {
    '__getitem__': _MT[tuple, tuple_getitem],
    '__hash__': _MT[tuple, tuple_hash],
    '__init__': _FT[tuple_init],
    '__len__': _MT[tuple, tuple_len],
}

_tuple_attrs = _tuple_methods.copy()
_tuple_attrs.update({
    '__bases__': _Tuple[_Ty[object]],
    '__name__': _Cst['tuple'],
})

def tuple_instanciate(ty):
    return _tuple_methods

##
#

def range_iterator_iter(self):
    return self

def range_iterator_next(self):
    return int

_range_iterator_attrs = {
    '__iter__': _MT[range_iterator, range_iterator_iter],
    '__next__': _MT[range_iterator, range_iterator_next],
}

##
#

def str_iterator_next(self):
    if self is str_iterator:
        return str
    elif isinstance(self, str_iterator):
        return str
    else:
        raise TypeError

_str_iterator_attrs = {
    '__next__': _FT[str_iterator_next],
}

##
#

def abs_(x):
    from penty.penty import Types
    return Types[x]['__abs__'](x)

##
#

def all_(iterable):
    from penty.penty import Types
    if '__len__' in Types[iterable]:
        len_ty = Types[iterable]['__len__'](iterable)
        if len_ty is _Cst[0]:
            return _Cst[True]

    elt_tys = set()
    iter_tys = _asset(iter_(iterable))
    for iter_ty in iter_tys:
        elt_tys.update(_asset(next_(iter_ty)))

    for elt_ty in elt_tys:
        if bool_init(elt_ty) is _Cst[False]:
            return _Cst[False]

    return bool


##
#

def any_(iterable):
    from penty.penty import Types
    if '__len__' in Types[iterable]:
        len_ty = Types[iterable]['__len__'](iterable)
        if len_ty is _Cst[0]:
            return _Cst[False]

    elt_tys = set()
    iter_tys = _asset(iter_(iterable))
    for iter_ty in iter_tys:
        elt_tys.update(_asset(next_(iter_ty)))

    for elt_ty in elt_tys:
        if bool_init(elt_ty) is _Cst[True]:
            return _Cst[True]

    return bool

##
#

def ascii_(obj):
    return str

##
#

def bin_(number):
    return hex_(number)

##
#

def callable_(obj):
    return _Cst[issubclass(obj, (_FDef, _Lambda, _FT, _Ty))]

##
#

def chr_(i):
    from penty.penty import Types
    if issubclass(i, int):
        return str
    if '__index__' in Types[i]:
        return chr_(Types[i]['__index__'](i))
    raise TypeError

##
#
def divmod_(x, y):
    from penty.penty import Types
    x, y = _astype(x), _astype(y)
    try:
        return Types[x]['__divmod__'](x, y)
    except TypeError:
        if '__rdivmod__' in Types[y]:
            return Types[y]['__rdivmod__'](y, x)
        raise
##
#

def enumerate_(iterable, start=int):
    from penty.penty import Types
    iterable = _astype(iterable)
    start = _astype(start)
    if not issubclass(start, int):
        if '__index__' in Types[start]:
            return enumerate_(iterable, Types[start]['__index__'](start))
        else:
            raise TypeError

    iter_ty = iter_(iterable)
    elts_ty = next_(iter_ty)
    return _Generator[{_Tuple[int, elt_ty] for elt_ty in elts_ty}]
##
#

def filter_(func, iterable):
    from penty.penty import Types
    typer_instance = _get_typer()
    iter_ty = iter_(iterable)
    elts_ty = _asset(next_(iter_ty))
    if func is _Cst[None]:
        return _Generator[elts_ty]
    else:
        results_ty = typer_instance._call(func, *elts_ty)
        for result_ty in results_ty:
            bool_ty = Types[bool]['__init__'](result_ty)
            if _astype(bool_ty) is not bool:
                raise TypeError
        return _Generator[elts_ty]
##
#

def getattr_(obj, name, default=None):
    from penty.penty import Types

    if name is str:
        raise NotImplementedError
    if not (issubclass(name, _Cst) and isinstance(name.__args__[0], str)):
        raise TypeError

    attr = name.__args__[0]
    if attr in Types[obj]:
        typer_instance = _get_typer()
        return typer_instance._handle_attr(obj, attr)
    elif default is not None:
        return default
    else:
        raise TypeError

##
#

def hasattr_(obj, name, node=None):
    from penty.penty import Types

    if name is str:
        return bool
    if not (issubclass(name, _Cst) and isinstance(name.__args__[0], str)):
        raise TypeError

    attr = name.__args__[0]
    if node is None:
        return _Cst[attr in Types[obj]]
    else:
        return _FilteringBool[attr in Types[obj], node.id, (obj,)]

##
#

def hash_(obj):
    from penty.penty import Types
    if '__hash__' not in Types[obj]:
        raise TypeError
    hash_ty = Types[obj]['__hash__']
    if hash_ty is _Cst[None]:
        raise TypeError
    return hash_ty(obj)

##
#

def hex_(number):
    from penty.penty import Types
    if issubclass(number, int):
        return str
    if '__index__' in Types[number]:
        return hex_(Types[number]['__index__'](number))
    raise TypeError

##
#

def id_(obj):
    return int

##
#

def input_(prompt=_Cst[None]):
    prompt = _astype(prompt)
    if issubclass(prompt, (str, _Cst[None])):
        return str
    raise TypeError

##
#

def repr_(obj):
    return str

##
#

def iter_(obj, sentinel=None):
    from penty.penty import Types
    if sentinel is None:
        if '__iter__' not in Types[obj]:
            raise TypeError
        return Types[obj]['__iter__'](obj)
    else:
        typer_instance = _get_typer()
        result_tys = _asset(typer_instance._call(obj))
        if _astype(sentinel) not in result_tys:
            raise RuntimeError
        return _Generator[result_tys]

##
#

def len_(obj):
    from penty.penty import Types
    if issubclass(obj, _Cst):
        return _Cst[len(obj.__args__[0])]
    else:
        return Types[obj]['__len__'](obj)

##
#

def isinstance_(obj, class_or_tuple, node=None):
    from penty.penty import Types
    type_ty, = Types[_Module['builtins']]['type']
    obj = type_ty(obj, node)
    return issubclass_(obj, class_or_tuple)

##
#

def issubclass_(cls, class_or_tuple):
    orig_cls_ty = cls

    def helper(cls, class_ty, node):
        from penty.penty import Types
        from penty.pentypes.operator import IsOperator
        issame = IsOperator(cls, class_ty)
        if issame.__args__[0]:
            return _FilteringBool[True, node, (class_ty.__args__[0],)]

        if cls is _Ty[object]:
            return _FilteringBool[False, node, (orig_cls_ty.__args__[0],)]

        for base_ty in Types[cls]['__bases__'].__args__:
            isbasesame = helper(base_ty, class_ty, node)
            if isbasesame.__args__[0]:
                return _FilteringBool[True, node, (class_ty.__args__[0],)]

        return _FilteringBool[False, node, (orig_cls_ty.__args__[0],)]

    node = cls.__args__[1] if issubclass(cls, _TypeOf) else None

    if issubclass(class_or_tuple, tuple):
        if not class_or_tuple.__args__[0]:
            raise TypeError

        result_tys = {helper(cls, class_ty, node)
                      for class_ty in class_or_tuple.__args__}

        for result_ty in result_tys:
            if result_ty.__args__[0]:
                return result_ty
        return result_tys
    else:
        return helper(cls, class_or_tuple, node)

##
#

def reversed_(sequence):
    from penty.penty import Types
    if '__reversed__' not in Types[sequence]:
        raise TypeError
    return Types[sequence]['__reversed__'](sequence)

##
#

def round_(number, ndigits=None):
    from penty.penty import Types
    if ndigits and not issubclass(_astype(ndigits), int):
        raise TypeError

    if issubclass(number, (int, float)):
        if ndigits is None:
            return int
        else:
            return number
    return Types[number]['__round__'](number, ndigits)

##
#

def map_(func_ty, *args_ty):
    from penty.penty import Types
    typer_instance = _get_typer()
    iters_ty = [iter_(arg_ty) for arg_ty in args_ty]
    elts_ty = [_asset(next_(iter_ty)) for iter_ty in iters_ty]
    result_ty = typer_instance._call(func_ty, *elts_ty)
    return _Generator[result_ty]

##
#

def max_(arg1, *args, default=None, key=None):
    from penty.penty import Types
    typer_instance = _get_typer()
    # max of a sequence
    if not args:
        iters_ty = _asset(iter_(arg1))
        elts_ty = [_asset(next_(iter_ty)) for iter_ty in iters_ty]

        comparables = set()
        if key is not None:
            for elt_ty in  elts_ty:
                for ety in elt_ty:
                    comparables.update(_asset(typer_instance._call(key, ety)))
        else:
            comparables.update(*elts_ty)

        # check that elements are comparable
        for self, value in _itertools.product(comparables, comparables):
            Types[_Module['operator']]['gt'](self, value)

        if default is None:
            result_ty = set()
        else:
            result_ty = {_astype(default)}

        return result_ty.union(*elts_ty)

    # max among elements
    all_args = {_astype(arg1)}
    all_args.update(map(_astype, args))

    comparables = set()
    if key is not None:
        for arg in all_args:
            comparables.update(_asset(typer_instance._call(key, arg)))
    else:
        comparables.update(all_args)
    # check that elements are comparable
    for self, value in _itertools.product(comparables, comparables):
        Types[_Module['operator']]['gt'](self, value)

    return all_args

##
#

def min_(arg1, *args, default=None, key=None):
    return max_(arg1, *args, default=default, key=key)

##
#

def next_(iterator, default=None):
    from penty.penty import Types
    if '__next__' not in Types[iterator]:
        raise TypeError
    result_tys = _asset(Types[iterator]['__next__'](iterator))
    if default is not None:
        result_tys.add(_astype(default))
    return result_tys

##
#

def oct_(number):
    return hex_(number)

##
#

def ord_(c):
    if not issubclass(c, str):
        raise TypeError
    return int

##
#

def pow_impl(base, exp, mod):
    from penty.penty import Types
    base = _astype(base)
    exp = _astype(exp)
    mod = _astype(mod)
    if '__pow__' in Types[base]:
        return Types[base]['__pow__'](base, exp, mod)
    else:
        raise TypeError

if sys.version_info < (3, 8):
    def pow_(x, y, z=_Cst[None]):
        return pow_impl(x, y, z)
else:
    def pow_(base, exp, mod=_Cst[None]):
        return pow_impl(base, exp, mod)

##
#

text_io_wrapper = type(sys.stdout)

def print_(*values, sep=_Cst[' '], end=_Cst['\n'], file=text_io_wrapper, flush=_Cst[False]):
    sep = _astype(sep)
    if not (sep is _Cst[None] or sep is str):
        raise TypeError
    end = _astype(end)
    if not (end is _Cst[None] or end is str):
        raise TypeError

    from penty.penty import Types
    if 'write' not in Types[file]:
        raise TypeError

    bool_init(flush)
    return _Cst[None]

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

def sorted_(iterable, *, key=_Cst[None], reverse=_Cst[False]):
    from penty.penty import Types

    elts_ty = _asset(next_(iter_(iterable)))

    # check key argument
    typer_instance = _get_typer()
    if key is not _Cst[None]:
        keys_ty = set()
        for elt_ty in elts_ty:
            keys_ty.update(_asset(typer_instance._call(key, elt_ty)))
    else:
        keys_ty = elts_ty

    # check that elements are comparable
    for self, value in _itertools.product(keys_ty, keys_ty):
        Types[_Module['operator']]['lt'](self, value)

    # check reverse keyword argument
    if '__bool__' not in Types[_astype(reverse)]:
        raise TypeError

    return _List[elts_ty]

##
#

def sum_(iterable, start=_Cst[0]):
    from penty.penty import Types
    if '__len__' in Types[iterable]:
        len_ty = Types[iterable]['__len__'](iterable)
        if len_ty is _Cst[0]:
            return start

    result_tys = {_astype(start)}
    iter_tys = _asset(iter_(iterable))
    for iter_ty in iter_tys:
        result_tys.update(_asset(next_(iter_ty)))

    # check that elements are addable
    for self, value in _itertools.product(result_tys, result_tys):
        Types[_Module['operator']]['add'](self, value)

    return result_tys

##
#

def type_(self, node=None):
    if node is None:
        return _Ty[_astype(self)]
    else:
        return _TypeOf[_astype(self), node.id]

##
#

def zip_(*iters):
    from penty.penty import Types
    typer_instance = _get_typer()
    iters_ty = [iter_(i) for i in iters]
    elts_tys = [_asset(next_(iter_ty)) for iter_ty in iters_ty]
    return {_Generator[_Tuple[elts_ty]]
            for elts_ty in _itertools.product(*elts_tys)}

##
#

def register(registry):
    if _Module['builtins'] not in registry:
        # the registration order must respect the type hierarchy, top-down
        registry[object] = _object_attrs
        registry[complex] = resolve_base_attrs(_complex_attrs, registry)
        registry[int] = resolve_base_attrs(_int_attrs, registry)
        registry[bool] = resolve_base_attrs(_bool_attrs, registry)
        registry[dict] = resolve_base_attrs(_dict_attrs, registry)
        registry[list] = resolve_base_attrs(_list_attrs, registry)
        registry[float] = resolve_base_attrs(_float_attrs, registry)
        registry[range] = resolve_base_attrs(_range_attrs, registry)
        registry[set] = resolve_base_attrs(_set_attrs, registry)
        registry[str] = resolve_base_attrs(_str_attrs, registry)
        registry[tuple] = resolve_base_attrs(_tuple_attrs, registry)
        registry[type(None)] = resolve_base_attrs(_none_attrs, registry)
        registry[_FilteringBool] = _FilteringBool_attrs
        registry[_SetIterator] = set_iterator_instanciate
        registry[range_iterator] = _range_iterator_attrs
        registry[str_iterator] = _str_iterator_attrs
        registry[_Dict] = dict_instanciate
        registry[_DictItemIterator] = dict_item_iterator_instanciate
        registry[_DictKeyIterator] = dict_key_iterator_instanciate
        registry[_DictValueIterator] = dict_value_iterator_instanciate
        registry[_List] = list_instanciate
        registry[_ListIterator] = list_iterator_instanciate
        registry[_Set] = set_instanciate
        registry[_Tuple] = tuple_instanciate

        registry[_Module['builtins']] = {
            'abs': {_CFT[abs_, abs]},
            'all': {_CFT[all_, all]},
            'any': {_CFT[any_, any]},
            'ascii': {_CFT[ascii_, ascii]},
            'bin': {_CFT[bin_, bin]},
            'bool': {_Ty[bool]},
            # 'bytearray': {},
            # 'bytes': {},
            'callable': {_CFT[callable_, callable]},
            'chr': {_CFT[chr_, chr]},
            # 'classmethod': {},
            # 'compile': {},
            'complex': {_Ty[complex]},
            # 'copyright': {},
            # 'credits': {},
            # 'delattr': {},
            'dict': {_Ty[dict]},
            # 'dir': {},
            'divmod': {_CFT[divmod_, divmod]},
            'enumerate': {_FT[enumerate_]},
            # 'eval': {},
            # 'exec': {},
            # 'exit': {},
            'filter': {_CFT[filter_, filter]},
            'float': {_Ty[float]},
            # 'format': {},
            # 'frozenset': {},
            'getattr': {_CFT[getattr_, getattr]},
            # 'globals': {},
            'hasattr': {_CFT[hasattr_, hasattr]},
            'hash': {_CFT[hash_, hash]},
            # 'help': {},
            'hex': {_CFT[hex_, hex]},
            'id': {_FT[id_]},
            'input': {_FT[input_]},
            'int': {_Ty[int]},
            'isinstance': {_CFT[isinstance_, isinstance]},
            'issubclass': {_CFT[issubclass_, issubclass]},
            'iter': {_CFT[iter_, iter]},
            'len': {_CFT[len_, len]},
            # 'license': {},
            'list': {_Ty[list]},
            # 'locals': {},
            'map': {_FT[map_]},
            'max': {_CFT[max_, max]},
            # 'memoryview': {},
            'min': {_CFT[min_, min]},
            'next': {_FT[next_]},
            'object': {_Ty[object]},
            'oct': {_CFT[oct_, oct]},
            # 'open': {},
            'ord': {_CFT[ord_, ord]},
            'pow': {_CFT[pow_, pow]},
            'print': {_FT[print_]},
            # 'property': {},
            # 'quit': {},
            'range': {_Ty[range]},
            'repr': {_CFT[repr_, repr]},
            'reversed': {_FT[reversed_]},
            'round': {_CFT[round_, round]},
            'set': {_Ty[set]},
            # 'setattr': {},
            'slice': {_FT[slice_]},
            'sorted': {_FT[sorted_]},
            # 'staticmethod': {},
            'str': {_Ty[str]},
            'sum': {_CFT[sum_, sum]},
            # 'super': {},
            'tuple': {_Ty[tuple]},
            'type': {_FT[type_]},
            # 'vars': {},
            'zip': {_FT[zip_]},
        }
