from penty.pentypes.numpy import random

from penty.types import Module as _Module, Cst as _Cst, Type as _Ty
from penty.types import astype as _astype, FunctionType as _FT, Tuple as _Tuple
from penty.types import ConstFunctionType as _CFT, PropertyType as _PT
from penty.types import resolve_base_attrs

import itertools
import numbers
import numpy as _np
import operator

def _broadcast_dim(self, other):
    if self is other:
        return self
    if self is int or other is int:
        return int
    if issubclass(self, _Cst) and issubclass(other, _Cst):
        if self.__args__[0] == 1:
            return other
        if other.__args__[0] == 1:
            return self
        raise TypeError
    raise TypeError

def _broadcast_shape(self, other):
    return _Tuple[tuple(_broadcast_dim(*dims) for dims in
                               itertools.zip_longest(self.__args__, other.__args__,
                                                     fillvalue=_Cst[1]))]

#
##

_generic_attrs = {
    '__bases__' : _Tuple[_Ty[object]],
    'imag' : _PT[lambda self_ty: _Cst[self_ty(0)], None],
    'real' : _PT[lambda self_ty: self_ty, None],
    'itemsize': _PT[lambda self_ty: _Cst[self_ty(0).itemsize], None],
    'nbytes': _PT[lambda self_ty: _Cst[self_ty(0).nbytes], None],
    'ndim': _Cst[0],
    'shape': _Tuple[()],
    'size': _Cst[1],
    'strides': _Tuple[()],
}

#
##

_number_attrs = {
    '__bases__' : _Tuple[_Ty[_np.generic]],
}

#
##

def _integer_round(self_ty):
    if not issubclass(self_ty, _np.integer):
        raise TypeError
    return type(self_ty(1).__round__())

_integer_attrs = {
    '__bases__' : _Tuple[_Ty[_np.number]],
    '__round__' : _CFT[_integer_round, _np.integer.__round__],
    'denominator' : _Cst[1],
    'numerator' : _PT[lambda self_ty: self_ty, None],
}

#
##

_signedinteger_attrs = {
    '__bases__' : _Tuple[_Ty[_np.integer]],
}

#
##

_unsignedinteger_attrs = {
    '__bases__' : _Tuple[_Ty[_np.integer]],
}

#
##
def make_dtype(dtype):

    def dtype_abs(self_ty):
        if issubclass(self_ty, dtype):
            return dtype
        else:
            raise TypeError

    def dtype_make_binop(operator):
        def binop(self_ty, other_ty):
            self_ty, other_ty = _astype(self_ty), _astype(other_ty)
            if not issubclass(self_ty, dtype):
                raise TypeError
            if issubclass(other_ty, (_np.generic, int, float, complex)):
                return type(operator(self_ty(1), other_ty(1)))
            else:
                raise TypeError
        return _CFT[binop, operator]

    def dtype_make_rbinop(operator):
        def rbinop(self_ty, other_ty):
            self_ty, other_ty = _astype(self_ty), _astype(other_ty)
            if not issubclass(self_ty, dtype):
                raise TypeError
            if issubclass(other_ty, (_np.generic, int, float, complex)):
                return type(operator(other_ty(1), self_ty(1)))
            else:
                raise TypeError
        return _CFT[rbinop, lambda x, y: operator(y, x)]

    def dtype_make_bitop(operator):
        def binop(self_ty, other_ty):
            self_ty, other_ty = _astype(self_ty), _astype(other_ty)
            if not issubclass(self_ty, dtype):
                raise TypeError
            if issubclass(other_ty, (_np.generic, int)):
                try:
                    return type(operator(self_ty(1), other_ty(1)))
                except TypeError:
                    raise TypeError  # FIXME: for a better message
            else:
                raise TypeError
        return _CFT[binop, operator]

    def dtype_make_unaryop(operator):
        def unaryop(self_ty):
            if not issubclass(self_ty, dtype):
                raise TypeError
            return type(operator(self_ty(1)))
        return _CFT[unaryop, operator]

    def dtype_bool(self_ty):
        if issubclass(self_ty, dtype):
            return bool
        else:
            raise TypeError

    def dtype_divmod(self_ty, other_ty):
        self_ty, other_ty = _astype(self_ty), _astype(other_ty)
        if not issubclass(self_ty, dtype):
            raise TypeError
        if issubclass(other_ty, (_np.generic, int, float, complex)):
            tmp = divmod(self_ty(), other_ty(1))
            return _Tuple[type(tmp[0]), type(tmp[1])]
        else:
            raise TypeError

    def dtype_eq(self_ty, other_ty):
        self_ty = _astype(self_ty)
        if not issubclass(self_ty, dtype):
            raise TypeError
        return bool

    def dtype_float(self_ty):
        if issubclass(self_ty, dtype):
            return float
        else:
            raise TypeError

    def dtype_index(self_ty):
        if issubclass(self_ty, dtype):
            return int
        else:
            raise TypeError

    def dtype_init(value_ty=None):
        from penty.penty import Types
        if value_ty is None:
            return _Cst[dtype()]
        if '__int__' not in Types[_astype(value_ty)]:
            raise TypeError
        return dtype

    def dtype_int(self_ty):
        if issubclass(self_ty, dtype):
            return int
        else:
            raise TypeError

    def dtype_ne(self_ty, other_ty):
        self_ty = _astype(self_ty)
        if not issubclass(self_ty, dtype):
            raise TypeError
        return bool

    def dtype_rdivmod(self_ty, other_ty):
        self_ty, other_ty = _astype(self_ty), _astype(other_ty)
        if not issubclass(self_ty, dtype):
            raise TypeError
        if issubclass(other_ty, (_np.generic, int, float, complex)):
            tmp = divmod(other_ty(1), self_ty(1))
            return _Tuple[type(tmp[0]), type(tmp[1])]
        else:
            raise TypeError

    def dtype_str(self_ty):
        if not issubclass(self_ty, dtype):
            raise TypeError
        return str

    _dtype_attrs = {
        '__abs__': _CFT[dtype_abs, dtype.__abs__],
        '__add__': dtype_make_binop(operator.add),
        '__and__': dtype_make_bitop(operator.and_),
        '__bases__': _Tuple[_Ty[dtype.__bases__[0]]],
        '__bool__': _CFT[dtype_bool, dtype.__bool__],
        '__divmod__': _CFT[dtype_divmod, dtype.__divmod__],
        '__eq__': _CFT[dtype_eq, dtype.__eq__],
        '__float__': _CFT[dtype_float, dtype.__float__],
        '__floordiv__': dtype_make_binop(operator.floordiv),
        '__ge__': dtype_make_binop(operator.ge),
        '__gt__': dtype_make_binop(operator.gt),
        '__index__': _CFT[dtype_index, dtype.__index__],
        '__init__': _CFT[dtype_init, dtype],
        '__int__': _CFT[dtype_int, dtype.__int__],
        '__invert__': dtype_make_unaryop(operator.invert),
        '__le__': dtype_make_binop(operator.le),
        '__lshift__': dtype_make_bitop(operator.lshift),
        '__lt__': dtype_make_binop(operator.lt),
        '__mod__': dtype_make_binop(operator.mod),
        '__mul__': dtype_make_binop(operator.mul),
        '__ne__': _CFT[dtype_ne, dtype.__ne__],
        '__neg__': dtype_make_unaryop(operator.neg),
        '__or__': dtype_make_bitop(operator.or_),
        '__pos__': dtype_make_unaryop(operator.pos),
        '__pow__': dtype_make_binop(operator.pow),
        '__radd__': dtype_make_rbinop(operator.add),
        '__rand__': dtype_make_rbinop(operator.and_),
        '__rdivmod__': dtype_divmod,
        '__rfloordiv__': dtype_make_rbinop(operator.floordiv),
        '__rlshift__': dtype_make_rbinop(operator.lshift),
        '__rmod__': dtype_make_rbinop(operator.mod),
        '__rmul__': dtype_make_rbinop(operator.mul),
        '__ror__': dtype_make_rbinop(operator.or_),
        '__rpow__': dtype_make_rbinop(operator.pow),
        '__rrshift__': dtype_make_rbinop(operator.rshift),
        '__rshift__': dtype_make_bitop(operator.rshift),
        '__rsub__': dtype_make_rbinop(operator.sub),
        '__rtruediv__': dtype_make_rbinop(operator.truediv),
        '__rxor__': dtype_make_rbinop(operator.xor),
        '__str__': _CFT[dtype_str, dtype.__str__],
        '__sub__': dtype_make_binop(operator.sub),
        '__truediv__': dtype_make_binop(operator.truediv),
        '__xor__': dtype_make_bitop(operator.xor),
    }

    return _dtype_attrs

_int8_attrs = make_dtype(_np.int8)
_uint8_attrs = make_dtype(_np.uint8)
_int16_attrs = make_dtype(_np.int16)
_uint16_attrs = make_dtype(_np.uint16)
_int32_attrs = make_dtype(_np.int32)
_uint32_attrs = make_dtype(_np.uint32)
_int64_attrs = make_dtype(_np.int64)
_uint64_attrs = make_dtype(_np.uint64)

#
##

class NDArrayMeta(type):
    cache = {}
    def __getitem__(self, args):
        if args not in NDArrayMeta.cache:
            class LocalNDArray(NDArray):
                __args__ = args
            NDArrayMeta.cache[args] = LocalNDArray
        return NDArrayMeta.cache[args]

    def __repr__(self):
        return 'NDArray[{}]'.format(', '.join(map(str, self.__args__)))


class NDArray(_np.ndarray, metaclass=NDArrayMeta):
    pass


def ndarray_make_binop(op):
    def binop(self_ty, other_ty):
        from penty.penty import Types
        dtype_ty, shape_ty = self_ty.__args__
        other_ty =_astype(other_ty)
        if other_ty in (bool, int, float):
            new_dtype_ty = Types[_Module['operator']][op](dtype_ty, other_ty)
            return NDArray[new_dtype_ty, shape_ty]
        if issubclass(other_ty, NDArray):
            other_dtype_ty, other_shape_ty = other_ty.__args__
            new_dtype_ty = Types[_Module['operator']][op](dtype_ty,
                                                          other_dtype_ty)
            return NDArray[new_dtype_ty,
                           _broadcast_shape(shape_ty, other_shape_ty)]
        raise TypeError
    return _FT[binop]

def ndarray_make_unaryop(op):
    def unaryop(self_ty):
        from penty.penty import Types
        dtype_ty, shape_ty = self_ty.__args__
        return NDArray[Types[dtype_ty][op](dtype_ty), shape_ty]
    return _FT[unaryop]

def ndarray_make_bitop(op):
    def binop(self_ty, other_ty):
        from penty.penty import Types
        dtype_ty, shape_ty = self_ty.__args__
        if dtype_ty not in (bool, int):
            raise TypeError
        other_ty =_astype(other_ty)
        if other_ty in (bool, int):
            new_dtype_ty = Types[_Module['operator']][op](dtype_ty, other_ty)
            return NDArray[new_dtype_ty, shape_ty]
        if issubclass(other_ty, NDArray):
            other_dtype_ty, other_shape_ty = other_ty.__args__
            if other_dtype_ty not in (bool, int):
                raise TypeError
            new_dtype_ty = Types[_Module['operator']][op](dtype_ty,
                                                          other_dtype_ty)
            return NDArray[new_dtype_ty,
                           _broadcast_shape(shape_ty, other_shape_ty)]
        raise TypeError
    return _FT[binop]

def ndarray_invert(self_ty):
    from penty.penty import Types
    dtype_ty, shape_ty = self_ty.__args__
    if dtype_ty not in (bool, int):
        raise TypeError
    return NDArray[Types[dtype_ty]['__invert__'](dtype_ty), shape_ty]

def ndarray_matmul(self_ty, other_ty):
    from penty.penty import Types
    dtype_ty, shape_ty = self_ty.__args__
    other_ty =_astype(other_ty)
    if other_ty in (bool, int, float):
        raise TypeError
    if issubclass(other_ty, NDArray):
        other_dtype_ty, other_shape_ty = other_ty.__args__
        # using mul instead of matmul for type inference as matmul is not
        # defined for some scalars, including int
        new_dtype_ty = Types[_Module['operator']]['__mul__'](dtype_ty,
                                                             other_dtype_ty)
        return NDArray[new_dtype_ty,
                       _broadcast_shape(shape_ty, other_shape_ty)]
    raise TypeError

def ndarray_getitem(self_ty, key_ty):
    dtype_ty, shape_ty= self_ty.__args__

    if _astype(key_ty) is int:
        if len(shape_ty.__args__) == 1:
            return dtype_ty
        else:
            return NDArray[dtype_ty,
                           _Tuple[shape_ty.__args__[1:]]]
    if _astype(key_ty) is slice:
        return ndarray_getitem(self_ty, _Tuple[key_ty])

    if issubclass(_astype(key_ty), tuple):
        if len(shape_ty.__args__) < len(key_ty.__args__):
            raise TypeError
        new_shape = ()
        padded_indices_dims = itertools.zip_longest(key_ty.__args__,
                                                    shape_ty.__args__,
                                                    fillvalue=slice)
        for index, dim in padded_indices_dims:
            if index is int:
                continue
            if index is slice:
                new_shape += _astype(dim),
                continue
            if issubclass(index, _Cst):
                index = index.__args__[0]
                if isinstance(index, int):
                    continue
                if isinstance(index, slice):
                    if issubclass(dim, _Cst):
                        dim_v = dim.__args__[0]
                        new_shape += _Cst[len([0] * dim_v)[index]]
                    elif index.stop is None or index.stop < 0:
                        new_shape += int,
                    else:
                        start = 0 if index.start is None else index.start
                        step = 1 if index.step is None else index.step
                        if start < 0:
                            new_shape += int,
                        else:
                            new_shape += _Cst[(index.stop - start) // step],
                    continue
                raise TypeError
            raise TypeError
        if new_shape:
            return NDArray[dtype_ty, _Tuple[new_shape]]
        else:
            return dtype_ty
    raise TypeError(key_ty)

def ndarray_len(self_ty):
    _, shape_ty = self_ty.__args__
    return shape_ty.__args__[0]

def ndarray_str(self_ty):
    return str

def ndarray_bool(self_ty):
    _, shape_ty = self_ty.__args__
    dims = shape_ty.__args__
    if all((issubclass(d, _Cst) and d.__args__[0]) for d in dims):
        return _Cst[True]
    if any((issubclass(d, _Cst) and not d.__args__[0]) for d in dims):
        return _Cst[False]
    return bool

#
##


def ones_(shape_ty, dtype_ty=None):
    if dtype_ty is None:
        dtype_ty = float
    elif issubclass(dtype_ty, _Ty):
        dtype_ty = dtype_ty.__args__[0]
    else:
        raise NotImplementedError
    if shape_ty is int or issubclass(shape_ty, _Cst):
        return NDArray[dtype_ty, _Tuple[shape_ty]]
    if issubclass(shape_ty, _Tuple):
        return NDArray[dtype_ty, shape_ty]
    raise NotImplementedError


def ndarray_instanciate(ty):
    return {
        '__abs__': _FT[ndarray_make_unaryop('__abs__')],
        '__add__': _FT[ndarray_make_binop('__add__')],
        '__and__': _FT[ndarray_make_bitop('__and__')],
        '__bool__': _FT[ndarray_bool],
        '__eq__': _FT[ndarray_make_binop('__eq__')],
        '__floordiv__': _FT[ndarray_make_binop('__floordiv__')],
        '__ge__': _FT[ndarray_make_binop('__ge__')],
        '__gt__': _FT[ndarray_make_binop('__gt__')],
        '__getitem__': _FT[ndarray_getitem],
        '__invert__': _FT[ndarray_invert],
        '__le__': _FT[ndarray_make_binop('__le__')],
        '__len__': _FT[ndarray_len],
        '__lshift__': _FT[ndarray_make_bitop('__lshift__')],
        '__lt__': _FT[ndarray_make_binop('__lt__')],
        '__matmul__': _FT[ndarray_matmul],
        '__mod__': _FT[ndarray_make_binop('__mod__')],
        '__mul__': _FT[ndarray_make_binop('__mul__')],
        '__ne__': _FT[ndarray_make_binop('__ne__')],
        '__neg__': _FT[ndarray_make_unaryop('__neg__')],
        '__or__': _FT[ndarray_make_bitop('__or__')],
        '__pos__': _FT[ndarray_make_unaryop('__pos__')],
        '__pow__': _FT[ndarray_make_binop('__pow__')],
        '__rshift__': _FT[ndarray_make_bitop('__rshift__')],
        '__str__': _FT[ndarray_str],
        '__sub__': _FT[ndarray_make_binop('__sub__')],
        '__truediv__': _FT[ndarray_make_binop('__truediv__')],
        '__xor__': _FT[ndarray_make_bitop('__xor__')],
        'dtype': _Ty[ty.__args__[0]],
        'ndim': _Cst[len(ty.__args__[1].__args__)],
        'shape': ty.__args__[1],
    }

_ndarray_attrs = {
    '__name__' : _Cst['ndarray'],
    '__bases__': _Tuple[_Ty[object]],
}

#
##

def register(registry):
    if _Module['numpy'] not in registry:
        registry[_np.ndarray] = ndarray_instanciate
        registry[NDArray] = ndarray_instanciate
        registry[_np.generic] = resolve_base_attrs(_generic_attrs, registry)
        registry[_np.number] = resolve_base_attrs(_number_attrs, registry)
        registry[_np.integer] = resolve_base_attrs(_integer_attrs, registry)
        registry[_np.signedinteger] = resolve_base_attrs(_signedinteger_attrs,
                                                         registry)
        registry[_np.unsignedinteger] = resolve_base_attrs(_unsignedinteger_attrs,
                                                         registry)
        registry[_np.int8] = resolve_base_attrs(_int8_attrs, registry)
        registry[_np.uint8] = resolve_base_attrs(_uint8_attrs, registry)
        registry[_np.int16] = resolve_base_attrs(_int16_attrs, registry)
        registry[_np.uint16] = resolve_base_attrs(_uint16_attrs, registry)
        registry[_np.int32] = resolve_base_attrs(_int32_attrs, registry)
        registry[_np.uint32] = resolve_base_attrs(_uint32_attrs, registry)
        registry[_np.int64] = resolve_base_attrs(_int64_attrs, registry)
        registry[_np.uint64] = resolve_base_attrs(_uint64_attrs, registry)

        registry[_Module['numpy']] = {
            'int8': _Ty[_np.int8],
            'uint8': _Ty[_np.uint8],
            'int16': _Ty[_np.int16],
            'uint16': _Ty[_np.uint16],
            'int32': _Ty[_np.int32],
            'uint32': _Ty[_np.uint32],
            'int64': _Ty[_np.int64],
            'uint64': _Ty[_np.uint64],
            'ones': _FT[ones_],
            'random': _Module['numpy.random'],
        }
