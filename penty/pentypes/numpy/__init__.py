from penty.pentypes.numpy import random

from penty.pentypes import builtins as penbuiltins
from penty.types import Module as _Module, Cst as _Cst, Type as _Type
from penty.types import astype as _astype
import typing as _typing
import numpy
import itertools
import operator as _operator

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
    return _typing.Tuple[tuple(_broadcast_dim(*dims) for dims in
                               itertools.zip_longest(self.__args__, other.__args__,
                                                     fillvalue=_Cst[1]))]

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



class NDArray(metaclass=NDArrayMeta):
    pass

def ndarray_make_binop(op):
    def binop(self_ty, other_ty):
        from penty.penty import Types
        dtype_ty, shape_ty = self_ty.__args__
        other_ty =_astype(other_ty)
        if other_ty in (bool, int, float):
            return NDArray[Types[dtype_ty][op](dtype_ty, other_ty), shape_ty]
        if issubclass(other_ty, NDArray):
            other_dtype_ty, other_shape_ty = other_ty.__args__
            return NDArray[Types[dtype_ty][op](dtype_ty, other_dtype_ty),
                           _broadcast_shape(shape_ty, other_shape_ty)]
        raise TypeError
    return _Cst[binop]

def ndarray_make_unaryop(op):
    def unaryop(self_ty):
        from penty.penty import Types
        dtype_ty, shape_ty = self_ty.__args__
        return NDArray[Types[dtype_ty][op](dtype_ty), shape_ty]
    return _Cst[unaryop]

def ndarray_make_bitop(op):
    def binop(self_ty, other_ty):
        from penty.penty import Types
        dtype_ty, shape_ty = self_ty.__args__
        if dtype_ty not in (bool, int):
            raise TypeError
        other_ty =_astype(other_ty)
        if other_ty in (bool, int):
            return NDArray[Types[dtype_ty][op](dtype_ty, other_ty), shape_ty]
        if issubclass(other_ty, NDArray):
            other_dtype_ty, other_shape_ty = other_ty.__args__
            if other_dtype_ty not in (bool, int):
                raise TypeError
            return NDArray[Types[dtype_ty][op](dtype_ty, other_dtype_ty),
                           _broadcast_shape(shape_ty, other_shape_ty)]
        raise TypeError
    return _Cst[binop]

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
        return NDArray[Types[dtype_ty]['__mul__'](dtype_ty, other_dtype_ty),
                       _broadcast_shape(shape_ty, other_shape_ty)]
    raise TypeError

def ndarray_getitem(base_ty, self_ty, key_ty):
    from penty.penty import Types
    dtype_ty, shape_ty= base_ty.__args__

    if _astype(key_ty) is int:
        if len(shape_ty.__args__) == 1:
            return dtype_ty
        else:
            return NDArray[dtype_ty,
                           _typing.Tuple[shape_ty.__args__[1:]]]
    if _astype(key_ty) is slice:
        return ndarray_getitem(base_ty, self_ty, _typing.Tuple[key_ty])

    if issubclass(_astype(key_ty), tuple) :
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
            return NDArray[dtype_ty, _typing.Tuple[new_shape]]
        else:
            return dtype_ty
    raise TypeError(key_ty)

def ndarray_len(base_ty, self_ty):
    _, shape_ty = base_ty.__args__
    return shape_ty.__args__[0]

def ndarray_str(base_ty, self_ty):
    return str

def ndarray_dtype(base_ty, self_ty):
    return self_ty.__args__[0]

def ndarray_bool(base_ty, self_ty):
    _, shape_ty = base_ty.__args__
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
    elif issubclass(dtype_ty, _Type):
        dtype_ty = dtype_ty.__args__[0]
    else:
        raise NotImplementedError
    if shape_ty is int or issubclass(shape_ty, _Cst):
        return NDArray[dtype_ty, _typing.Tuple[shape_ty]]
    if issubclass(shape_ty, _typing.Tuple):
        return NDArray[dtype_ty, shape_ty]
    raise NotImplementedError

def ndarray_instanciate(ty):
    return {
        '__abs__': _Cst[ndarray_make_unaryop('__abs__')],
        '__add__': _Cst[ndarray_make_binop('__add__')],
        '__and__': _Cst[ndarray_make_bitop('__and__')],
        '__bool__': _Cst[lambda *args: ndarray_bool(ty, *args)],
        '__eq__': _Cst[ndarray_make_binop('__eq__')],
        '__floordiv__': _Cst[ndarray_make_binop('__floordiv__')],
        '__ge__': _Cst[ndarray_make_binop('__ge__')],
        '__gt__': _Cst[ndarray_make_binop('__gt__')],
        '__getitem__': _Cst[lambda *args: ndarray_getitem(ty, *args)],
        '__invert__': _Cst[ndarray_invert],
        '__le__': _Cst[ndarray_make_binop('__le__')],
        '__len__': _Cst[lambda *args: ndarray_len(ty, *args)],
        '__lshift__': _Cst[ndarray_make_bitop('__lshift__')],
        '__lt__': _Cst[ndarray_make_binop('__lt__')],
        '__matmul__': _Cst[ndarray_matmul],
        '__mod__': _Cst[ndarray_make_binop('__mod__')],
        '__mul__': _Cst[ndarray_make_binop('__mul__')],
        '__ne__': _Cst[ndarray_make_binop('__ne__')],
        '__neg__': _Cst[ndarray_make_unaryop('__neg__')],
        '__or__': _Cst[ndarray_make_bitop('__or__')],
        '__pos__': _Cst[ndarray_make_unaryop('__pos__')],
        '__pow__': _Cst[ndarray_make_binop('__pow__')],
        '__rshift__': _Cst[ndarray_make_bitop('__rshift__')],
        '__str__': _Cst[lambda *args: ndarray_str(ty, *args)],
        '__sub__': _Cst[ndarray_make_binop('__sub__')],
        '__truediv__': _Cst[ndarray_make_binop('__truediv__')],
        '__xor__': _Cst[ndarray_make_bitop('__xor__')],
        'dtype': _Type[ty.__args__[0]],
    }

#
##

def register(registry):
    if _Module['numpy'] not in registry:
        registry[NDArray] = ndarray_instanciate
        registry[_Module['numpy']] = {
            'ones': _Cst[ones_],
            'random': _Module['numpy.random'],
        }
