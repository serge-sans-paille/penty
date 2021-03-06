from unittest import TestCase

import builtins
import gast as ast
import penty.types as pentyping
import math
import numpy as np
import typing

from pentest import TestPenty, inject_spec_test

class TestSpecs(TestCase):
    pass

inject_spec_test(TestSpecs, pentyping.Module['builtins'], builtins, 'builtins')


class TestBuiltins(TestPenty):

    def test_abs(self):
        self.assertIsType('abs(x)', int, env={'x': bool})
        self.assertIsType('abs(x)', int, env={'x': int})
        self.assertIsType('abs(x)', float, env={'x': float})

    def test_all(self):
        self.assertIsType('all([])', pentyping.Cst[True])
        self.assertIsType('all("er")', pentyping.Cst[True])
        self.assertIsType('all(x)', bool, env={'x': str})
        self.assertIsType('all(x)', pentyping.Cst[False],
                          env={'x': pentyping.Tuple[pentyping.Cst[1],
                                                    pentyping.Cst[""]]})

    def test_any(self):
        self.assertIsType('any([])', pentyping.Cst[False])
        self.assertIsType('any("er")', pentyping.Cst[True])
        self.assertIsType('any(x)', bool, env={'x': str})
        self.assertIsType('any(x)', pentyping.Cst[True],
                          env={'x': pentyping.Tuple[pentyping.Cst[1],
                                                    pentyping.Cst[""]]})

    def test_ascii(self):
        self.assertIsType('ascii(3)', pentyping.Cst[ascii(3)])
        self.assertIsType('ascii("er")', pentyping.Cst[ascii("er")])
        self.assertIsType('ascii([])', str)
        self.assertIsType('ascii(())', str)
        self.assertIsType('ascii(x)', str, env={'x': str})

    def test_bin(self):
        self.assertIsType('bin(3)', pentyping.Cst[bin(3)])
        self.assertIsType('bin(x)', str, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('bin(x)', None, env={'x': complex})

    def test_callable(self):
        self.assertIsType('callable(1)', pentyping.Cst[False])
        self.assertIsType('callable(str)', pentyping.Cst[callable(str)])
        self.assertIsType('callable(lambda x:1)',
                          pentyping.Cst[callable(lambda x:1)])
        self.assertIsType('callable(int.__add__)',
                          pentyping.Cst[callable(int.__add__)])
        self.assertIsType('callable(abs)', pentyping.Cst[callable(abs)])
        self.assertIsType('callable(type(x))',
                          pentyping.Cst[callable(type("e"))],
                          env={'x': str})

    def test_chr(self):
        self.assertIsType('chr(31)', pentyping.Cst[chr(31)])
        self.assertIsType('chr(x)', str, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('chr(x)', None, env={'x': str})
        with self.assertRaises(TypeError):
            self.assertIsType('chr("x")', None)

    def test_complex(self):
        self.assertIsType('complex(x)', complex, env={'x': bool})
        self.assertIsType('complex(x)', complex, env={'x': int})
        self.assertIsType('complex(x)', complex, env={'x': float})
        self.assertIsType('complex(x)', complex, env={'x': complex})
        self.assertIsType('complex(x)', complex, env={'x': str})

    def test_enumerate(self):
        self.assertIsType('enumerate(x)',
                          pentyping.Generator[pentyping.Tuple[int, int]],
                          env={'x': pentyping.List[int]})
        self.assertIsType('enumerate(x, 3)',
                          pentyping.Generator[pentyping.Tuple[int, int]],
                          env={'x': pentyping.List[int]})
        self.assertIsType('enumerate("x", 3)',
                          pentyping.Generator[pentyping.Tuple[int, str]])
        self.assertIsType('enumerate(x)',
                          pentyping.Generator[{pentyping.Tuple[int, int],
                                               pentyping.Tuple[int, float]}],
                          env={'x': pentyping.List[{int, float}]})

    def test_hasattr(self):
        self.assertIsType('hasattr(x, "__add__")',
                          pentyping.FilteringBool[True, 'x', (bool,)],
                          env={'x': bool})
        self.assertIsType('hasattr(x, "__fly__")',
                          pentyping.FilteringBool[False, 'x', (bool,)],
                          env={'x': bool})
        self.assertIsType('hasattr(x, y)', bool,
                          env={'x': bool, 'y': str})
        self.assertIsType('hasattr(1, y)', bool,
                          env={'y': str})
        self.assertIsType('hasattr(1, "__add__")',
                          pentyping.Cst[True])

    def test_input(self):
        self.assertIsType('input()', str)
        self.assertIsType('input(None)', str)
        self.assertIsType('input(x)', str, env={'x': str})
        with self.assertRaises(TypeError):
            self.assertIsType('input(x)', None, env={'x': int})

    def test_getattr(self):
        self.assertIsType('getattr(x, "__add__")(1)',
                          int,
                          env={'x': bool})
        self.assertIsType('getattr(x, "__sadd__", lambda x: 0)(1)',
                          pentyping.Cst[0],
                          env={'x': bool})
        self.assertIsType('getattr(True, "__add__")(1)',
                          pentyping.Cst[True + 1])
        self.assertIsType('getattr(type(x), "__add__")(False, 1)',
                          pentyping.Cst[False + 1],
                          env={'x': bool})
        with self.assertRaises(TypeError):
            self.assertIsType('getattr(x, "__pub__")', None,
                              env={'x': bool})

    def test_print(self):
        self.assertIsType('print(x)', pentyping.Cst[None], env={'x': int})
        self.assertIsType('print(x, sep="e")', pentyping.Cst[None], env={'x': int})
        self.assertIsType('print(x, end="e")', pentyping.Cst[None], env={'x': int})
        self.assertIsType('print(x, flush=True)', pentyping.Cst[None], env={'x': int})

    def test_range(self):
        with self.assertRaises(TypeError):
            self.assertIsType('range()', None)
        self.assertIsType('range(3)', range)
        self.assertIsType('range(x)', range, env={'x': int})
        self.assertIsType('range(x, 3)', range, env={'x': int})
        self.assertIsType('range(x, 3, x)', range, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('range(x, None, x)', range, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('range(x, 3, 1)', range, env={'x': float})

    def test_round(self):
        self.assertIsType('round(3)', pentyping.Cst[round(3)])
        self.assertIsType('round(x)', int, env={'x': int})
        self.assertIsType('round(x)', int, env={'x': float})
        self.assertIsType('round(x, 2)', int, env={'x': int})
        self.assertIsType('round(x, 2)', float, env={'x': float})
        with self.assertRaises(TypeError):
            self.assertIsType('round(x, 2.)', None, env={'x': float})

    def test_str(self):
        self.assertIsType('str(x)', str, env={'x': bool})
        self.assertIsType('str(x)', str, env={'x': int})
        self.assertIsType('str(x)', str, env={'x': float})
        self.assertIsType('str(x)', str, env={'x': complex})
        self.assertIsType('str(x)', str, env={'x': str})

    def test_bool(self):
        self.assertIsType('bool(True)', pentyping.Cst[True])
        self.assertIsType('bool(False)', pentyping.Cst[False])
        self.assertIsType('bool(x)', bool, env={'x': bool})
        self.assertIsType('bool(x)', bool,
                          env={'x': int})
        self.assertIsType('bool(x)', bool,
                          env={'x': float})
        self.assertIsType('bool(x)', pentyping.Cst[True],
                          env={'x': pentyping.Tuple[int, int]})
        self.assertIsType('bool(2)', pentyping.Cst[True])
        self.assertIsType('bool(x)', bool,
                          env={'x': pentyping.List[int]})
        self.assertIsType('bool(x)', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('bool(x)', bool,
                          env={'x': pentyping.Dict[int, str]})
        self.assertIsType('bool(x)', bool,
                          env={'x': str})
        self.assertIsType('bool("")', pentyping.Cst[False])
        self.assertIsType('bool("hello")', pentyping.Cst[True])

    def test_float(self):
        self.assertIsType('float(x)', float, env={'x': bool})
        self.assertIsType('float(x)', float, env={'x': int})
        self.assertIsType('float(x)', float, env={'x': float})
        self.assertIsType('float(x)', float, env={'x': str})

    def test_hash(self):
        self.assertIsType('hash(True)', pentyping.Cst[hash(True)])
        self.assertIsType('hash(3)', pentyping.Cst[hash(3)])
        self.assertIsType('hash(3.)', pentyping.Cst[hash(3.)])
        self.assertIsType('hash(3j)', pentyping.Cst[hash(3j)])
        self.assertIsType('hash("3")', pentyping.Cst[hash("3")])
        self.assertIsType('hash(x)', int, env={'x': bool})
        self.assertIsType('hash(x)', int, env={'x': int})
        self.assertIsType('hash(x)', int, env={'x': float})
        self.assertIsType('hash(x)', int, env={'x': complex})
        self.assertIsType('hash(x)', int, env={'x': str})
        self.assertIsType('hash(x)', int, env={'x': pentyping.Tuple[int]})
        with self.assertRaises(TypeError):
            self.assertIsType('hex(x)', None, env={'x': pentyping.Set[int]})
        with self.assertRaises(TypeError):
            self.assertIsType('hex(x)', None, env={'x': pentyping.List[int]})

    def test_hex(self):
        self.assertIsType('hex(3)', pentyping.Cst[hex(3)])
        self.assertIsType('hex(x)', str, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('hex(x)', None, env={'x': complex})

    def test_int(self):
        self.assertIsType('int(x)', int, env={'x': bool})
        self.assertIsType('int(x)', int, env={'x': int})
        self.assertIsType('int(x)', int, env={'x': float})
        self.assertIsType('int(x)', int, env={'x': str})

    def test_isinstance(self):
        self.assertIsType('isinstance(1., int)', pentyping.Cst[False])
        self.assertIsType('isinstance(1, int)', pentyping.Cst[True])
        self.assertIsType('isinstance(True, int)', pentyping.Cst[True])
        self.assertIsType('isinstance(x, int)',
                          pentyping.FilteringBool[True, 'x', (int,)],
                          env={'x': int})
        self.assertIsType('isinstance(x, float)',
                          pentyping.FilteringBool[False, 'x', (int,)],
                          env={'x': int})
        self.assertIsType('isinstance(x, (int, float))',
                          pentyping.FilteringBool[True, 'x', (int,)],
                          env={'x': int})
        self.assertIsType('isinstance(x, (int, float))',
                          pentyping.FilteringBool[True, 'x', (float,)],
                          env={'x': float})
        self.assertIsType('isinstance(x, (int, float))',
                          pentyping.FilteringBool[False, 'x', (str,)],
                          env={'x': str})
        self.assertIsType('isinstance(x, (int, float))',
                          pentyping.FilteringBool[False, 'x', (pentyping.Cst[None],)],
                          env={'x': pentyping.Cst[None]})
        self.assertIsType('isinstance(x, (int, type(None)))',
                          pentyping.FilteringBool[True, 'x', (pentyping.Cst[None],)],
                          env={'x': pentyping.Cst[None]})

    def test_issubclass(self):
        self.assertIsType('issubclass(int, float)', pentyping.Cst[False])
        self.assertIsType('issubclass(int, (float, bool))', pentyping.Cst[False])
        self.assertIsType('issubclass(bool, int)', pentyping.Cst[True])
        self.assertIsType('issubclass(bool, object)', pentyping.Cst[True])
        self.assertIsType('issubclass(bool, (float, object))', pentyping.Cst[True])
        self.assertIsType('issubclass(x, float)', pentyping.Cst[True],
                          env={"x": pentyping.Type[float]})
        self.assertIsType('issubclass(x, float)', {pentyping.Cst[True],
                                                   pentyping.Cst[False]},
                          env={"x": {pentyping.Type[float], pentyping.Type[int]}})
        self.assertIsType('issubclass(type(x), float)',
                          pentyping.FilteringBool[True, 'x', (float,)],
                          env={"x": float})
        self.assertIsType('issubclass(type(x), float)',
                          pentyping.FilteringBool[False, 'x', (int,)],
                          env={"x": int})
        self.assertIsType('issubclass(type(x), int)',
                          pentyping.FilteringBool[True, 'x', (int,)],
                          env={"x": bool})
        self.assertIsType('issubclass(type(x), (float, int))',
                          pentyping.FilteringBool[True, 'x', (int,)],
                          env={"x": bool})
        self.assertIsType('issubclass(type(x), (float, int))',
                          pentyping.FilteringBool[False, 'x', (str,)],
                          env={"x": str})

    def test_len(self):
        self.assertIsType('len(x)', pentyping.Cst[2],
                          env={'x': pentyping.Tuple[int, int]})
        self.assertIsType('len(x)', int,
                          env={'x': pentyping.List[int]})
        self.assertIsType('len(x)', int,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('len(x)', int,
                          env={'x': pentyping.Dict[int, str]})
        self.assertIsType('len(x)', int,
                          env={'x': str})
        self.assertIsType('len("hello")', pentyping.Cst[5])

    def test_map(self):
        self.assertIsType('map(len, "1")', pentyping.Generator[int])
        self.assertIsType('map(len, x)', pentyping.Generator[int],
                          env={'x': str})
        self.assertIsType('map(lambda y: 1., x)',
                          pentyping.Generator[pentyping.Cst[1.]],
                          env={'x': str})
        self.assertIsType('map(float, x)', pentyping.Generator[float],
                          env={'x': str})
        self.assertIsType('map(lambda a, b: a or b, x, y)',
                          pentyping.Generator[str],
                          env={'x': str, 'y': str})

    def test_max(self):
        self.assertIsType('max(1, 2)', pentyping.Cst[max(1,2)])
        self.assertIsType('max(1, 2.)', pentyping.Cst[max(1, 2.)])
        self.assertIsType('max(x, 2.)', {int, float}, env={'x': int})
        self.assertIsType('max(x, x + 1, 2.)', {int, float}, env={'x': int})
        self.assertIsType('max([1, 2, 3])', int)
        self.assertIsType('max([x, 2, 3])', {int, float}, env={'x': float})
        self.assertIsType('max([x, "2", "3"], key=int)', str, env={'x': str})
        self.assertIsType('max(x)', bool, env={'x': pentyping.List[bool]})
        self.assertIsType('max(x, default=1)', {int, bool}, env={'x': pentyping.List[bool]})
        self.assertIsType('max(x, key=lambda x: -x)', bool, env={'x': pentyping.List[bool]})

    def test_min(self):
        self.assertIsType('min(1, 2)', pentyping.Cst[min(1,2)])
        self.assertIsType('min(1, 2.)', pentyping.Cst[min(1, 2.)])
        self.assertIsType('min(x, 2.)', {int, float}, env={'x': int})
        self.assertIsType('min(x, x + 1, 2.)', {int, float}, env={'x': int})
        self.assertIsType('min([1, 2, 3])', int)
        self.assertIsType('min([x, 2, 3])', {int, float}, env={'x': float})
        self.assertIsType('min([x, "2", "3"], key=int)', str, env={'x': str})
        self.assertIsType('min(x)', bool, env={'x': pentyping.List[bool]})
        self.assertIsType('min(x, default=1)', {int, bool}, env={'x': pentyping.List[bool]})
        self.assertIsType('min(x, key=lambda x: -x)', bool, env={'x': pentyping.List[bool]})

    def test_oct(self):
        self.assertIsType('oct(3)', pentyping.Cst[oct(3)])
        self.assertIsType('oct(x)', str, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('oct(x)', None, env={'x': complex})

    def test_ord(self):
        self.assertIsType('ord("3")', pentyping.Cst[ord("3")])
        self.assertIsType('ord(x)', int, env={'x': str})
        with self.assertRaises(TypeError):
            self.assertIsType('ord(x)', None, env={'x': complex})

    def test_pow(self):
        self.assertIsType('pow(3, 4.)', pentyping.Cst[pow(3, 4.)])
        self.assertIsType('pow(x, 4.)', float, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('pow(x, 2)', None, env={'x': pentyping.Set[int]})

    def test_sorted(self):
        self.assertIsType('sorted(x)',
                          pentyping.List[str],
                          env={'x': str})
        self.assertIsType('sorted(x, key=int, reverse=True)',
                          pentyping.List[str],
                          env={'x': str})
        self.assertIsType('sorted(x, reverse=True, key=int)',
                          pentyping.List[str],
                          env={'x': str})

    def test_sum(self):
        self.assertIsType('sum([])', pentyping.Cst[0])
        self.assertIsType('sum(x)', pentyping.Cst[5],
                          env={'x': pentyping.Tuple[pentyping.Cst[3], pentyping.Cst[2]]})
        self.assertIsType('sum({}, start=3)', pentyping.Cst[3])
        with self.assertRaises(TypeError):
            self.assertIsType('sum("hello")', None)
        self.assertIsType('sum(x, start="")', str,
                          env={'x': str})
        self.assertIsType('sum(x)', {int, float},
                          env={'x': pentyping.List[float]})

    def test_tuple(self):
        self.assertIsType('tuple()', pentyping.Tuple[()])
        self.assertIsType('tuple("er")', pentyping.Tuple[str, str])
        self.assertIsType('tuple(x)', pentyping.Tuple[str, int],
                          env={'x': pentyping.Tuple[str, int]})
        self.assertIsType('tuple(x)', pentyping.Tuple[int, ...],
                          env={'x': pentyping.List[int]})
        self.assertIsType('tuple([])', pentyping.Tuple[()])
        with self.assertRaises(TypeError):
            self.assertIsType('tuple(x)', tuple, env={'x': int})

    def test_type(self):
        self.assertIsType('type(x) is int',
                          pentyping.FilteringBool[True, 'x', (int,)],
                          env={'x':int})
        self.assertIsType('abs(x) if type(x) is int else x',
                          {int, pentyping.Cst[None]},
                          env={'x': {int, pentyping.Cst[None]}})
        self.assertIsType('type(x)(x)', int, env={'x': int})

    def test_reversed(self):
        self.assertIsType('reversed(x)', pentyping.ListIterator[{int, float}],
                          env={'x': pentyping.List[{int, float}]})

    def test_zip(self):
        self.assertIsType('zip("2", "1")',
                          pentyping.Generator[pentyping.Tuple[str, str]])
        self.assertIsType('zip(x, "1", y)',
                          pentyping.Generator[pentyping.Tuple[int, str, bool]],
                          env={'x': pentyping.List[int],
                               'y': pentyping.Set[bool]})
        self.assertIsType('zip(x, "1")',
                          {pentyping.Generator[pentyping.Tuple[int, str]],
                           pentyping.Generator[pentyping.Tuple[str, str]]},
                          env={'x': pentyping.List[{int,str}]})



class TestBool(TestPenty):

    def test_abs(self):
        self.assertIsType('abs(True)', pentyping.Cst[abs(True)])
        self.assertIsType('abs(x)', int, env={'x': bool})

    def test_add(self):
        self.assertIsType('True + False', pentyping.Cst[True + False])
        self.assertIsType('x + 1', int, env={'x': bool})
        self.assertIsType('x + 1.', float, env={'x': bool})

    def test_and(self):
        self.assertIsType('True & False', pentyping.Cst[True & False])
        self.assertIsType('False & 5', pentyping.Cst[False & 5])
        self.assertIsType('x & x', bool, env={'x': bool})
        self.assertIsType('x & 1', int, env={'x': bool})

    def test_bool(self):
        self.assertIsType('bool(True)', pentyping.Cst[bool(True)])
        self.assertIsType('bool(x)', bool, env={'x': bool})

    def test_or(self):
        self.assertIsType('True | False', pentyping.Cst[True | True])
        self.assertIsType('False | 5', pentyping.Cst[False | 5])
        self.assertIsType('x | x', bool, env={'x': bool})
        self.assertIsType('x | 1', int, env={'x': bool})

    def test_xor(self):
        self.assertIsType('True ^ False', pentyping.Cst[True ^ False])
        self.assertIsType('False ^ 5', pentyping.Cst[False ^ 5])
        self.assertIsType('x ^ x', bool, env={'x': bool})
        self.assertIsType('x ^ 1', int, env={'x': bool})

class TestInt(TestPenty):

    def test_abs(self):
        self.assertIsType('abs(1)', pentyping.Cst[abs(1)])
        self.assertIsType('abs(x)', int, env={'x': int})

    def test_ceil(self):
        self.assertIsType('int.__ceil__(1)', pentyping.Cst[math.ceil(1)])
        self.assertIsType('int.__ceil__(x)', int, env={'x': int})

    def test_bit_length(self):
        self.assertIsType('int.bit_length(1)', pentyping.Cst[int.bit_length(1)])
        self.assertIsType('x.bit_length()', int, env={'x': int})

    def test_conjugate(self):
        self.assertIsType('int.conjugate(1)', pentyping.Cst[int.conjugate(1)])
        self.assertIsType('x.conjugate()', int, env={'x': int})

    def test_denominator(self):
        self.assertIsType('x.denominator', pentyping.Cst[1], env={'x': int})

    def test_init(self):
        self.assertIsType('int()', pentyping.Cst[0])
        self.assertIsType('int(x)', int, env={'x': int})
        self.assertIsType('int(x)', int, env={'x': float})
        self.assertIsType('int(x)', int, env={'x': str})
        self.assertIsType('int(x, 4)', int, env={'x': str})

    def test_imag(self):
        self.assertIsType('x.imag', pentyping.Cst[0], env={'x': int})

    def test_numerator(self):
        self.assertIsType('x.numerator', int, env={'x': int})

    def test_pow(self):
        self.assertIsType('pow(3, 2)', pentyping.Cst[pow(3, 2)])
        self.assertIsType('pow(3, 2.)', pentyping.Cst[pow(3, 2.)])
        self.assertIsType('pow(3, x)', int, env={'x': int})
        self.assertIsType('pow(3, x)', float, env={'x': float})
        self.assertIsType('pow(3, x, 2)', int, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('pow(3, x, 2)', int, env={'x': float})

    def test_real(self):
        self.assertIsType('x.real', int, env={'x': int})


class TestFloat(TestPenty):

    def test_abs(self):
        self.assertIsType('abs(-1.)', pentyping.Cst[abs(-1.)])
        self.assertIsType('abs(x)', float, env={'x': float})

    def test_add(self):
        self.assertIsType('1. + 3.4', pentyping.Cst[1 + 3.4])
        self.assertIsType('x + 3.4', float, env={'x': float})
        self.assertIsType('x + x', float, env={'x': float})
        self.assertIsType('2 + x', float, env={'x': float})

    def test_bool(self):
        self.assertIsType('bool(1.)', pentyping.Cst[bool(1.)])
        self.assertIsType('bool(x)', bool, env={'x': float})

    def test_divmod(self):
        dm = divmod(5.3, 2.)
        self.assertIsType('divmod(5.3, 2.)',
                          pentyping.Tuple[pentyping.Cst[dm[0]],
                                          pentyping.Cst[dm[1]]])
        self.assertIsType('x.__divmod__(4.)',
                          pentyping.Tuple[float, float],
                          env={'x': float})
        self.assertIsType('x.__divmod__(4)',
                          pentyping.Tuple[float, float],
                          env={'x': float})
        self.assertIsType('y.__divmod__(x)',
                          pentyping.Tuple[float, float],
                          env={'x': float, 'y': int})

    def test_eq(self):
        self.assertIsType('1. == 3.4', pentyping.Cst[1 == 3.4])
        self.assertIsType('x == 3.4', bool, env={'x': float})
        self.assertIsType('x == x', bool, env={'x': float})

    def test_float(self):
        self.assertIsType('float(3.4)', pentyping.Cst[float(3.4)])
        self.assertIsType('float(3)', pentyping.Cst[float(3)])
        self.assertIsType('float(x)', float, env={'x': bool})
        self.assertIsType('float(x)', float, env={'x': float})

    def test_floordiv(self):
        self.assertIsType('1. // 3.4', pentyping.Cst[1 // 3.4])
        self.assertIsType('x // 3.4', float, env={'x': float})
        self.assertIsType('x // x', float, env={'x': float})
        self.assertIsType('2 // x', float, env={'x': float})

    def test_ge(self):
        self.assertIsType('1. >= 3.4', pentyping.Cst[1 >= 3.4])
        self.assertIsType('x >= 3.4', bool, env={'x': float})
        self.assertIsType('x >= x', bool, env={'x': float})

    def test_gt(self):
        self.assertIsType('1. > 3.4', pentyping.Cst[1 > 3.4])
        self.assertIsType('x > 3.4', bool, env={'x': float})
        self.assertIsType('x > x', bool, env={'x': float})

    def test_init(self):
        self.assertIsType('float(1.2)', pentyping.Cst[float(1.2)])
        self.assertIsType('float(x)', float, env={'x': bool})
        self.assertIsType('float(True)', pentyping.Cst[float(True)])
        self.assertIsType('float(x)', float, env={'x': int})
        self.assertIsType('float(3)', pentyping.Cst[float(3)])
        self.assertIsType('float(x)', float, env={'x': float})
        self.assertIsType('float(2.1)', pentyping.Cst[float(2.1)])
        self.assertIsType('float(x)', float, env={'x': str})
        self.assertIsType('float("3.14")', pentyping.Cst[float("3.14")])

    def test_int(self):
        self.assertIsType('int(1.2)', pentyping.Cst[int(1.2)])
        self.assertIsType('int(x)', int, env={'x': float})

    def test_le(self):
        self.assertIsType('1. <= 3.4', pentyping.Cst[1 <= 3.4])
        self.assertIsType('x <= 3.4', bool, env={'x': float})
        self.assertIsType('x <= x', bool, env={'x': float})

    def test_lt(self):
        self.assertIsType('1. < 3.4', pentyping.Cst[1 < 3.4])
        self.assertIsType('x < 3.4', bool, env={'x': float})
        self.assertIsType('x < x', bool, env={'x': float})

    def test_mul(self):
        self.assertIsType('1. * 3.4', pentyping.Cst[1 * 3.4])
        self.assertIsType('x * 3.4', float, env={'x': float})
        self.assertIsType('x * x', float, env={'x': float})
        self.assertIsType('2 * x', float, env={'x': float})

    def test_mod(self):
        self.assertIsType('1. % 3.4', pentyping.Cst[1 % 3.4])
        self.assertIsType('x % 3.4', float, env={'x': float})
        self.assertIsType('x % x', float, env={'x': float})
        self.assertIsType('2 % x', float, env={'x': float})

    def test_ne(self):
        self.assertIsType('1. != 3.4', pentyping.Cst[1 != 3.4])
        self.assertIsType('x != 3.4', bool, env={'x': float})
        self.assertIsType('x != x', bool, env={'x': float})

    def test_neg(self):
        self.assertIsType('-(1.4)', pentyping.Cst[-(1.4)])
        self.assertIsType('-x', float, env={'x': float})

    def test_pos(self):
        self.assertIsType('+(1.4)', pentyping.Cst[+(1.4)])
        self.assertIsType('+x', float, env={'x': float})

    def test_pow(self):
        self.assertIsType('1. ** 3.4', pentyping.Cst[1 ** 3.4])
        self.assertIsType('x ** 3.4', float, env={'x': float})
        self.assertIsType('x ** x', float, env={'x': float})
        self.assertIsType('2 ** x', float, env={'x': float})
        with self.assertRaises(TypeError):
            self.assertIsType('pow(x, 2, 2)', None, env={'x': float})

    def test_sub(self):
        self.assertIsType('1. - 3.4', pentyping.Cst[1 - 3.4])
        self.assertIsType('x - 3.4', float, env={'x': float})
        self.assertIsType('x - x', float, env={'x': float})
        self.assertIsType('1 - x', float, env={'x': float})

    def test_truediv(self):
        self.assertIsType('1. / 3.4', pentyping.Cst[1 / 3.4])
        self.assertIsType('x / 3.4', float, env={'x': float})
        self.assertIsType('x / x', float, env={'x': float})
        self.assertIsType('1 / x', float, env={'x': float})


class TestComplex(TestPenty):

    def test_abs(self):
        self.assertIsType('abs(2.j)', pentyping.Cst[abs(2.j)])
        self.assertIsType('abs(x)', complex, env={'x': complex})

    def test_add(self):
        self.assertIsType('2.j + 3j', pentyping.Cst[2.j + 3j])
        self.assertIsType('x + x', complex, env={'x': complex})
        self.assertIsType('x + 3', complex, env={'x': complex})
        self.assertIsType('3 + x', complex, env={'x': complex})
        self.assertIsType('x + 3.1', complex, env={'x': complex})
        self.assertIsType('3.1 + x', complex, env={'x': complex})

    def test_bool(self):
        self.assertIsType('bool(0.j)', pentyping.Cst[bool(0.j)])
        self.assertIsType('bool(x)', bool, env={'x': complex})

    def test_divmod(self):
        with self.assertRaises(TypeError):
            self.assertIsType('divmod(x, 2.j)', None, env={'x': complex})

    def test_filter(self):
        self.assertIsType('filter(None, x)', pentyping.Generator[str],
                          env={'x': str})
        self.assertIsType('filter(len, x)', pentyping.Generator[str],
                          env={'x': str})
        self.assertIsType('filter(lambda x: x == 1, x)',
                          pentyping.Generator[int],
                          env={'x': pentyping.List[int]})
        with self.assertRaises(TypeError):
            self.assertIsType('filter(None, x)', None, env={'x': complex})
        with self.assertRaises(TypeError):
            self.assertIsType('filter(x, "er")', None, env={'x': complex})

    def test_iter(self):
        self.assertIsType('iter(x)', pentyping.ListIterator[str],
                          env={'x': pentyping.List[str]})
        self.assertIsType('iter(x)', type(iter("")),
                          env={'x': str})
        self.assertIsType('iter(lambda: x, 1)',
                          pentyping.Generator[int],
                          env={'x': int})

    def test_eq(self):
        self.assertIsType('x == x', bool, env={'x': complex})
        self.assertIsType('x == 1', bool, env={'x': complex})
        self.assertIsType('x == 1.', bool, env={'x': complex})

    def test_floordiv(self):
        with self.assertRaises(TypeError):
            self.assertIsType('x // 2', None, env={'x': complex})

    def test_ge(self):
        with self.assertRaises(TypeError):
            self.assertIsType('x >= 2', None, env={'x': complex})

    def test_gt(self):
        with self.assertRaises(TypeError):
            self.assertIsType('x >= 2', None, env={'x': complex})

    def test_init(self):
        self.assertIsType('complex(2, 3)', pentyping.Cst[complex(2, 3)])
        self.assertIsType('complex(x)', complex, env={'x': complex})
        self.assertIsType('complex(x)', complex, env={'x': int})
        self.assertIsType('complex(x, 1.)', complex, env={'x': int})
        self.assertIsType('complex(x, 1.)', complex, env={'x': complex})
        self.assertIsType('complex(x, 1j)', complex, env={'x': int})
        self.assertIsType('complex(x, 1j)', complex, env={'x': complex})

    def test_int(self):
        with self.assertRaises(TypeError):
            self.assertIsType('int(x)', None, env={'x': complex})

    def test_le(self):
        with self.assertRaises(TypeError):
            self.assertIsType('x <= 2', None, env={'x': complex})

    def test_lt(self):
        with self.assertRaises(TypeError):
            self.assertIsType('x < 2', None, env={'x': complex})

    def test_mul(self):
        self.assertIsType('1j * 2j', pentyping.Cst[1j * 2j])
        self.assertIsType('x * x', complex, env={'x': complex})
        self.assertIsType('x * 1', complex, env={'x': complex})
        self.assertIsType('x * 1.', complex, env={'x': complex})

    def test_mod(self):
        with self.assertRaises(TypeError):
            self.assertIsType('x % 2', None, env={'x': complex})

    def test_ne(self):
        self.assertIsType('x != x', bool, env={'x': complex})
        self.assertIsType('x != 1', bool, env={'x': complex})
        self.assertIsType('x != 1.', bool, env={'x': complex})

    def test_neg(self):
        self.assertIsType('-(1j)', pentyping.Cst[-(1j)])
        self.assertIsType('-x', complex, env={'x': complex})

    def test_next(self):
        self.assertIsType('next(iter("str"))', str)
        self.assertIsType('next(iter(x))', int, env={'x': pentyping.List[int]})
        self.assertIsType('next(iter(x), 1.)', {int, float},
                          env={'x': pentyping.List[int]})
        self.assertIsType('next(iter(x), None)', {int, pentyping.Cst[None]},
                          env={'x': pentyping.Set[int]})
        with self.assertRaises(TypeError):
            self.assertIsType('next(x)', None, env={'x': complex})

    def test_pos(self):
        self.assertIsType('+(1j)', pentyping.Cst[+(1j)])
        self.assertIsType('+x', complex, env={'x': complex})

    def test_pow(self):
        self.assertIsType('(1j) ** 2', pentyping.Cst[(1j) ** 2])
        self.assertIsType('x ** 2', complex, env={'x': complex})
        self.assertIsType('x ** 3.1', complex, env={'x': complex})
        self.assertIsType('x ** 3j', complex, env={'x': complex})
        with self.assertRaises(TypeError):
            self.assertIsType('pow(x, 2, 2)', None, env={'x': complex})

    def test_str(self):
        self.assertIsType('str(1j)', pentyping.Cst[str(1j)])
        self.assertIsType('str(x)', str, env={'x': complex})
        self.assertIsType('str(x)', str, env={'x':  pentyping.List[{int}]})

    def test_sub(self):
        self.assertIsType('1j - 2j', pentyping.Cst[1j - 2j])
        self.assertIsType('x - x', complex, env={'x': complex})
        self.assertIsType('x - 1', complex, env={'x': complex})
        self.assertIsType('x - 1.', complex, env={'x': complex})

    def test_truediv(self):
        self.assertIsType('1j / 2j', pentyping.Cst[1j / 2j])
        self.assertIsType('x / x', complex, env={'x': complex})
        self.assertIsType('x / 1', complex, env={'x': complex})
        self.assertIsType('x / 1.', complex, env={'x': complex})


class TestDict(TestPenty):
    def test_bool(self):
        self.assertIsType('bool(x)', pentyping.Cst[False],
                          env={'x': pentyping.Dict[set(), set()]})
        self.assertIsType('bool(x)', bool,
                          env={'x': pentyping.Dict[int, int]})

    def test_clear(self):
        self.assertIsType('dict.clear(x)', pentyping.Cst[None],
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('x.clear()', pentyping.Cst[None],
                          env={'x': pentyping.Dict[int, int]})

    def test_contains(self):
        self.assertIsType('1 in x', pentyping.Cst[False],
                          env={'x': pentyping.Dict[set(), set()]})
        self.assertIsType('1 in x', bool,
                          env={'x': pentyping.Dict[int, int]})

    def test_copy(self):
        self.assertIsType('x.copy()', pentyping.Dict[set(), set()],
                          env={'x': pentyping.Dict[set(), set()]})
        self.assertIsType('x.copy()', pentyping.Dict[int, int],
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('(x.copy(), x.__setitem__(1, 3.))[0]', pentyping.Dict[int, int],
                          env={'x': pentyping.Dict[int, int]})

    def test_delitem(self):
        self.assertIsType('x.__delitem__(1)', pentyping.Cst[None],
                          env={'x': pentyping.Dict[int, int]})

    def test_eq(self):
        self.assertIsType('x == y', pentyping.Cst[False],
                          env={'x': pentyping.Dict[int, int],
                               'y': int})
        self.assertIsType('x == x', bool,
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('x == y', pentyping.Cst[False],
                          env={'x': pentyping.Dict[int, int],
                               'y': pentyping.Dict[set(), set()]})

    def test_hash(self):
        self.assertIsType('x.__hash__', pentyping.Cst[None],
                          env={'x': pentyping.Dict[int, int]})
        with self.assertRaises(TypeError):
            self.assertIsType('dict([([x], 1)])', None, env={'x': complex})

    def test_init(self):
        self.assertIsType('dict()', pentyping.Dict[set(), set()])
        self.assertIsType('dict(x)', pentyping.Dict[str, int],
                          env={'x': pentyping.Dict[str, int]})
        self.assertIsType('dict(x)', pentyping.Dict[str, int],
                          env={'x': pentyping.Set[pentyping.Tuple[str, int]]})

    def test_ne(self):
        self.assertIsType('x != y', pentyping.Cst[True],
                          env={'x': pentyping.Dict[int, int],
                               'y': int})
        self.assertIsType('x != x', bool,
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('x != y', pentyping.Cst[True],
                          env={'x': pentyping.Dict[int, int],
                               'y': pentyping.Dict[set(), set()]})

    def test_gt(self):
        self.assertIsType('x > x', bool,
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('x > y', bool,
                          env={'x': pentyping.Dict[int, int],
                               'y': pentyping.Dict[set(), set()]})

    def test_lt(self):
        self.assertIsType('x < x', bool,
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('x < y', bool,
                          env={'x': pentyping.Dict[int, int],
                               'y': pentyping.Dict[set(), set()]})

    def test_ge(self):
        self.assertIsType('x >= x', bool,
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('x >= y', bool,
                          env={'x': pentyping.Dict[int, int],
                               'y': pentyping.Dict[set(), set()]})

    def test_le(self):
        self.assertIsType('x <= x', bool,
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('x <= y', bool,
                          env={'x': pentyping.Dict[int, int],
                               'y': pentyping.Dict[set(), set()]})

    def test_iter(self):
        self.assertIsType('set(x)', pentyping.Set[int],
                          env={'x': pentyping.Dict[int, float]})

    def test_len(self):
        self.assertIsType('len(x)', pentyping.Cst[0],
                          env={'x': pentyping.Dict[set(), set()]})
        self.assertIsType('len(x)', int,
                          env={'x': pentyping.Dict[int, int]})

    def test_fromkeys(self):
        self.assertIsType('dict.fromkeys(x)',
                          pentyping.Dict[str, pentyping.Cst[None]],
                          env={'x': str})

    def test_get(self):
        self.assertIsType('x.get(1, y)', int,
                          env={'x': pentyping.Dict[int, int],
                               'y': int})
        self.assertIsType('x.get(1, 0.)', {int, pentyping.Cst[0.]},
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('x.get(1)', {int, pentyping.Cst[None]},
                          env={'x': pentyping.Dict[int, int]})

    def test_getitem(self):
        self.assertIsType('x[1]', int,
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('x[y]', str,
                          env={'x': pentyping.Dict[int, str],
                               'y': int})

    def test_items(self):
        self.assertIsType('set(x.items())',
                          pentyping.Set[pentyping.Tuple[bool, int]],
                          env={'x': pentyping.Dict[bool, int]})
        self.assertIsType('set(x.items())',
                          pentyping.Set[{pentyping.Tuple[int, int],
                                         pentyping.Tuple[int, bool]}],
                          env={'x': pentyping.Dict[int, {int, bool}]})

    def test_pop(self):
        self.assertIsType('x.pop(1, y)', int,
                          env={'x': pentyping.Dict[int, int],
                               'y': int})
        self.assertIsType('x.pop(1, 0.)', {int, pentyping.Cst[0.]},
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('x.pop(1)', {int, pentyping.Cst[None]},
                          env={'x': pentyping.Dict[int, int]})

    def test_popitem(self):
        self.assertIsType('x.popitem()', pentyping.Tuple[int, int],
                          env={'x': pentyping.Dict[int, int]}),
        self.assertIsType('x.popitem()', pentyping.Tuple[int, float],
                          env={'x': pentyping.Dict[int, float]})
        self.assertIsType('x.popitem()', {pentyping.Tuple[int, float],
                                          pentyping.Tuple[int, str]},
                          env={'x': pentyping.Dict[int, {float, str}]})

    def test_setdefault(self):
        self.assertIsType('x.setdefault(1, 0)', int,
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('x.setdefault(1)', {int, pentyping.Cst[None]},
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('(x.setdefault(1), x)[1]',
                          {pentyping.Dict[int, {int, pentyping.Cst[None]}]},
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('(x.setdefault(1, []).append(1.), x)[1]',
                          pentyping.Dict[int, {pentyping.List[{int, float}],
                                               pentyping.List[{float}]}],
                          env={'x': pentyping.Dict[int, pentyping.List[int]]})

    def test_update(self):
        self.assertIsType('x.update(x)', pentyping.Cst[None],
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('(x.update(x), x)[1]', pentyping.Dict[int, int],
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('(x.update({(1, 0.)}), x)[1]',
                          pentyping.Dict[int, {int, float}],
                          env={'x': pentyping.Dict[int, int]})
        self.assertIsType('(x.update({(1, 0.)}, {(1., 0)}), x)[1]',
                          pentyping.Dict[{int, float}, {int, float}],
                          env={'x': pentyping.Dict[int, int]})

    def test_values(self):
        self.assertIsType('set(x.values())', pentyping.Set[int],
                          env={'x': pentyping.Dict[bool, int]})
        self.assertIsType('set(x.values())', pentyping.Set[{int, bool}],
                          env={'x': pentyping.Dict[int, {int, bool}]})

    def test_keys(self):
        self.assertIsType('set(x.keys())', pentyping.Set[bool],
                          env={'x': pentyping.Dict[bool, int]})
        self.assertIsType('set(x.keys())', pentyping.Set[int],
                          env={'x': pentyping.Dict[int, {int, bool}]})


class TestList(TestPenty):

    def test_add(self):
        self.assertIsType(
            'x + y', pentyping.List[{int}],
            env={'x': pentyping.List[{int}],
                 'y': pentyping.List[{int}]}
        )
        self.assertIsType(
            'x + y', pentyping.List[{int, float}],
            env={'x': pentyping.List[{int}],
                 'y': pentyping.List[{float}]}
        )
        with self.assertRaises(TypeError):
            self.assertIsType('x + y', None, env={'x': pentyping.List[{int}],
                                                  'y': int})

    def test_bool(self):
        self.assertIsType('bool(list())', pentyping.Cst[False], env={})
        self.assertIsType('bool(x)', bool, env={'x': pentyping.List[{int}]})

    def test_contains(self):
        self.assertIsType(
            'x in y',
            bool,
            env={'x': int, 'y': pentyping.List[{float}]}
        )

    def test_delitem(self):
        self.assertIsType(
            'x.__delitem__(y)',
            pentyping.Cst[None],
            env={'x': pentyping.List[{int}], 'y': int}
        )
        self.assertIsType(
            'x.__delitem__(y)',
            pentyping.Cst[None],
            env={'x': pentyping.List[{int}], 'y': pentyping.Cst[3]}
        )
        with self.assertRaises(TypeError):
            self.assertIsType(
                'x.__delitem__(y)',
                pentyping.Cst[None],
                env={'x': pentyping.List[{int}], 'y': float}
            )

    def test_eq(self):
        self.assertIsType(
            'x == y', bool, env={'x': pentyping.List[{int}],
                                 'y': pentyping.List[{int}]}
        )
        self.assertIsType(
            'x == 3', pentyping.Cst[False], env={'x': pentyping.List[{int}]}
        )

    def test_ge(self):
        self.assertIsType(
            'x >= y', bool, env={'x': pentyping.List[{int}],
                                 'y': pentyping.List[{int}]}
        )

    def test_getitem(self):
        self.assertIsType(
            'x[0]', {int, float}, env={'x': pentyping.List[{int, float}]}
        )
        self.assertIsType(
            'x[:3]', pentyping.List[{int, float}],
            env={'x': pentyping.List[{int, float}]}
        )

    def test_hash(self):
        self.assertIsType('x.__hash__', pentyping.Cst[None],
                          env={'x': pentyping.List[int]})

    def test_iadd(self):
        self.assertIsType(
            'x.__iadd__(y)', pentyping.List[{int, float}],
            env={'x': pentyping.List[{int}],
                 'y': pentyping.List[{float}]}
        )

    def test_imul(self):
        self.assertIsType(
            'x.__imul__(y)', pentyping.List[{int, float}],
            env={'x': pentyping.List[{int, float}], 'y': int}
        )
        with self.assertRaises(TypeError):
            self.assertIsType(
                'x.__imul__(list())', None,
                env={'x': pentyping.List[{int, float}]}
        )

    def test_le(self):
        self.assertIsType(
            'x <= y', bool, env={'x': pentyping.List[{int}],
                                 'y': pentyping.List[{int}]}
        )

    def test_len(self):
        self.assertIsType(
            'len(x)', int, env={'x': pentyping.List[{int}]}
        )
        self.assertIsType(
            'len(x)', pentyping.Cst[0], env={'x': pentyping.List[set()]}
        )
        self.assertIsType('len(list())', pentyping.Cst[0], env={})

    def test_lt(self):
        self.assertIsType(
            'x < y', bool, env={'x': pentyping.List[{int}],
                                'y': pentyping.List[{int}]}
        )

    def test_mul(self):
        self.assertIsType(
            'x * y', pentyping.List[{int, float}],
            env={'x': pentyping.List[{int, float}], 'y': int}
        )
        with self.assertRaises(TypeError):
            self.assertIsType(
                'x * list()', None, env={'x': pentyping.List[{int, float}]}
        )

    def test_ne(self):
        self.assertIsType('x != y', bool,
                          env={'x': pentyping.List[{int}],
                               'y': pentyping.List[{int, float}]})
        self.assertIsType('x != list()', bool,
                          env={'x': pentyping.List[{int}]})
        self.assertIsType('x != 3', pentyping.Cst[True],
                          env={'x': pentyping.List[{int}]})

    def test_repr(self):
        self.assertIsType('repr(x)', str,
                          env={'x': pentyping.List[{int}]})
        self.assertIsType('repr(list())', str, env={})

    def test_reversed(self):
        self.assertIsType('reversed(x)', pentyping.ListIterator[{float}],
                          env={'x': pentyping.List[{float}]})

    def test_rmul(self):
        self.assertIsType('x * y', pentyping.List[{int, float}],
                          {'x': int, 'y': pentyping.List[{int, float}]})

    def test_setitem(self):
        self.assertIsType(
            'x.__setitem__(y, 3), x',
            pentyping.Tuple[pentyping.Cst[None], pentyping.List[{int, float}]],
            env={'x': pentyping.List[{float}], 'y': int}
        )
        self.assertIsType(
            'x.__setitem__(y, [1, 2]), x',
            pentyping.Tuple[pentyping.Cst[None], pentyping.List[{int, float}]],
            env={'x': pentyping.List[{float}], 'y': slice}
        )
        with self.assertRaises(TypeError):
            self.assertIsType(
                'x.__setitem__(y, 2), x',
                None,
                env={'x': pentyping.List[{float}], 'y': slice}
            )

    def test_str(self):
        self.assertIsType('[1, 2].__str__()', str, env={})

    def test_clear(self):
        self.assertIsType('x.clear(), x', 
                          pentyping.Tuple[pentyping.Cst[None],
                                          pentyping.List[{int, float}]],
                          env={'x': pentyping.List[{int, float}]})

    def test_copy(self):
        self.assertIsType('x.copy(), x.append(y), x',
                          pentyping.Tuple[
                              pentyping.List[{float}],
                              pentyping.Cst[None],
                              pentyping.List[{int, float}]
                          ],
                          env={'x': pentyping.List[{float}],
                               'y': int})

    def test_count(self):
        self.assertIsType('x.count(y)', pentyping.Cst[0],
                          env={'x': pentyping.List[set()], 'y': int})
        self.assertIsType('x.count(y)', int,
                          env={'x': pentyping.List[{int}], 'y': int})

    def test_extend(self):
        self.assertIsType('x.extend(y), x',
                          pentyping.Tuple[pentyping.Cst[None],
                                          pentyping.List[{int}]],
                          env={'x': pentyping.List[set()],
                               'y': pentyping.List[{int}]})
        self.assertIsType('x.extend(y), x',
                          pentyping.Tuple[pentyping.Cst[None],
                                          pentyping.List[{int, float}]],
                          env={'x': pentyping.List[{float}],
                               'y': pentyping.List[{int}]})
        with self.assertRaises(TypeError):
            self.assertIsType('x.extend(y), x',
                              None,
                              env={'x': pentyping.List[{float}],
                                   'y': int})

    def test_index(self):
        self.assertIsType('x.index(y)',
                          int,
                          env={'x': pentyping.List[{int}], 'y': float})
        self.assertIsType('x.index(y, 0, 3)',
                          int,
                          env={'x': pentyping.List[{int}], 'y': float})
        with self.assertRaises(TypeError):
            self.assertIsType('x.index(y, 0.0, 3)',
                              None,
                              env={'x': pentyping.List[{int}], 'y': float})
        with self.assertRaises(TypeError):
            self.assertIsType('x.index(y, 0, 3.0)',
                              None,
                              env={'x': pentyping.List[{int}], 'y': float})
    def test_insert(self):
        self.assertIsType('x.insert(0, y), x',
                          pentyping.Tuple[pentyping.Cst[None],
                                          pentyping.List[{int, float}]],
                          env={'x': pentyping.List[{int}],
                               'y': float})
        with self.assertRaises(TypeError):
            self.assertIsType('x.insert(slice(), y), x',
                              None,
                              env={'x': pentyping.List[{int}],
                                   'y': float})
        with self.assertRaises(TypeError):
            self.assertIsType('x.insert(0.0, y), x',
                              None,
                              env={'x': pentyping.List[{int}],
                                   'y': float})

    def test_pop(self):
        self.assertIsType('x.pop()', {int}, env={'x': pentyping.List[{int}]})
        with self.assertRaises(TypeError):
            self.assertIsType('x.pop(y)', None,
                              env={'x': pentyping.List[{int}],
                                   'y': float})

    def test_reverse(self):
        self.assertIsType('x.reverse(), x',
                          pentyping.Tuple[pentyping.Cst[None],
                                          pentyping.List[{int}]],
                          env={'x': pentyping.List[{int}]})

    def test_append(self):
        self.assertIsType('x.append(y), x',
                          pentyping.Tuple[pentyping.Cst[None], pentyping.List[int]],
                          env={'x': pentyping.List[set()], 'y': int})
        self.assertIsType('(x.append(y), x)[1]',
                          {pentyping.List[{int, float}]},
                          env={'x': pentyping.List[float], 'y': int})

    def test_sort(self):
        self.assertIsType(
            'x.sort()',
            pentyping.Cst[None],
            env={'x': pentyping.List[{float}]})
        self.assertIsType(
            'x.sort(reverse=y)',
            pentyping.Cst[None],
            env={'x': pentyping.List[{float}],
                 'y': bool})
        self.assertIsType(
            'x.sort(key=int)',
            pentyping.Cst[None],
            env={'x': pentyping.List[{float}]})
        self.assertIsType(
            'x.sort(key=lambda a: a[0])',
            pentyping.Cst[None],
            env={'x': pentyping.List[{pentyping.Tuple[float, int]}]})
        with self.assertRaises(TypeError):
            self.assertIsType(
                'x.sort(reverse=y)',
                pentyping.Cst[None],
                env={'x': pentyping.List[{float}],
                     'y': float})


class TestSet(TestPenty):

    def test_bool(self):
        self.assertIsType('bool(x)', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('bool(x)',
                          pentyping.Cst[False],
                          env={'x': pentyping.Set[set()]})

    def test_and(self):
        self.assertIsType('x & x', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x & y', pentyping.Set[{int, float}],
                          env={'x': pentyping.Set[int],
                               'y': pentyping.Set[float]
                              })

    def test_contains(self):
        self.assertIsType('1 in x', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('1. in x', bool,
                          env={'x': pentyping.Set[int]})

    def test_eq(self):
        self.assertIsType('1 == x', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x == 1', pentyping.Cst[False],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x == x', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x == y', bool,
                          env={'x': pentyping.Set[int],
                               'y': pentyping.Set[str]})

    def test_ge(self):
        self.assertIsType('x >= x', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x >= y', bool,
                          env={'x': pentyping.Set[int],
                               'y': pentyping.Set[str]})

    def test_gt(self):
        self.assertIsType('x > x', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x > y', bool,
                          env={'x': pentyping.Set[int],
                               'y': pentyping.Set[str]})

    def test_hash(self):
        self.assertIsType('x.__hash__', pentyping.Cst[None],
                          env={'x': pentyping.Set[int]})
        with self.assertRaises(TypeError):
            self.assertIsType('set([[x]])', None, env={'x': complex})

    def test_init(self):
        self.assertIsType('set("ert")', pentyping.Set[str])

    def test_iter(self):
        self.assertIsType('[x for x in {1}]',
                          pentyping.List[int])
        self.assertIsType('{x for x in x}',
                          pentyping.Set[int],
                          env={'x': pentyping.Set[int]})

    def test_le(self):
        self.assertIsType('x <= x', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x <= y', bool,
                          env={'x': pentyping.Set[int],
                               'y': pentyping.Set[str]})

    def test_len(self):
        self.assertIsType('len(x)', int,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('len(x)', pentyping.Cst[0],
                          env={'x': pentyping.Set[set()]})

    def test_lt(self):
        self.assertIsType('x < x', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x < y', bool,
                          env={'x': pentyping.Set[int],
                               'y': pentyping.Set[str]})

    def test_ne(self):
        self.assertIsType('1 != x', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x != 1', pentyping.Cst[True],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x != x', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x != y', bool,
                          env={'x': pentyping.Set[int],
                               'y': pentyping.Set[str]})

    def test_or(self):
        self.assertIsType('x | x', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x | y', pentyping.Set[{int, float}],
                          env={'x': pentyping.Set[int],
                               'y': pentyping.Set[float]
                              })

    def test_sub(self):
        self.assertIsType('x - x', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x - y', pentyping.Set[int],
                          env={'x': pentyping.Set[int],
                               'y': pentyping.Set[float]
                              })

    def test_xor(self):
        self.assertIsType('x ^ x', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x ^ y', pentyping.Set[{int, float}],
                          env={'x': pentyping.Set[int],
                               'y': pentyping.Set[float]
                              })
    def test_add(self):
        self.assertIsType('x.add(1)', pentyping.Cst[None],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('(x.add(1), x)[1]', pentyping.Set[{int, float}],
                          env={'x': pentyping.Set[float]})

    def test_clear(self):
        self.assertIsType('x.clear()', pentyping.Cst[None],
                          env={'x': pentyping.Set[int]})

    def test_copy(self):
        self.assertIsType('x.copy()', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('(x.copy().add(1.), x)[1]', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})

    def test_difference(self):
        self.assertIsType('x.difference()', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.difference({1})', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.difference({1}, "er")', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})

    def test_difference_update(self):
        self.assertIsType('x.difference_update()', pentyping.Cst[None],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.difference_update("er")', pentyping.Cst[None],
                          env={'x': pentyping.Set[int]})

    def test_discard(self):
        self.assertIsType('x.discard(1)', pentyping.Cst[None],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.discard("1")', pentyping.Cst[None],
                          env={'x': pentyping.Set[int]})

    def test_intersection(self):
        self.assertIsType('x.intersection()', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.intersection({1})', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.intersection({1}, "er")',
                          pentyping.Set[{int, str}],
                          env={'x': pentyping.Set[int]})

    def test_intersection_update(self):
        self.assertIsType('x.intersection_update()', pentyping.Cst[None],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('(x.intersection_update({1}, "er"), x)[1]',
                          pentyping.Set[{int, str}],
                          env={'x': pentyping.Set[int]})

    def test_isdisjoint(self):
        self.assertIsType('x.isdisjoint(x)', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.isdisjoint("er")',
                          bool,
                          env={'x': pentyping.Set[int]})

    def test_issubset(self):
        self.assertIsType('x.issubset(x)', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.issubset("er")',
                          bool,
                          env={'x': pentyping.Set[int]})

    def test_issuperset(self):
        self.assertIsType('x.issuperset(x)', bool,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.issuperset("er")',
                          bool,
                          env={'x': pentyping.Set[int]})

    def test_pop(self):
        self.assertIsType('x.pop()', int,
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.pop()',
                          {bool, int},
                          env={'x': pentyping.Set[{bool, int}]})

    def test_remove(self):
        self.assertIsType('x.remove(1)', pentyping.Cst[None],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.remove("er")',
                          pentyping.Cst[None],
                          env={'x': pentyping.Set[{bool, int}]})

    def test_symmetric_difference(self):
        self.assertIsType('x.symmetric_difference({1})', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.symmetric_difference("1")',
                          pentyping.Set[{int,str}],
                          env={'x': pentyping.Set[int]})

    def test_symmetric_difference_update(self):
        self.assertIsType('x.symmetric_difference_update("1")',
                          pentyping.Cst[None],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('(x.symmetric_difference_update("1"), x)[1]',
                          pentyping.Set[{int, str}],
                          env={'x': pentyping.Set[int]})

    def test_union(self):
        self.assertIsType('x.union()', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.union({1})', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('set.union(x, {1})', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('x.union({1}, "er")',
                          pentyping.Set[{int, str}],
                          env={'x': pentyping.Set[int]})

    def test_update(self):
        self.assertIsType('x.update()', pentyping.Cst[None],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('(x.update({1}), x)[1]', pentyping.Set[int],
                          env={'x': pentyping.Set[int]})
        self.assertIsType('(x.update({1.}, "er"), x)[1]',
                          pentyping.Set[{int, float, str}],
                          env={'x': pentyping.Set[int]})


class TestRange(TestPenty):

    def test_bool(self):
        self.assertIsType('bool(range(x))', bool, env={'x': int})

    def test_contains(self):
        self.assertIsType('1 in range(x)', bool, env={'x': int})

    def test_hash(self):
        self.assertIsType('hash(range(x))', int, env={'x': int})

    def test_len(self):
        self.assertIsType('len(range(x))', int, env={'x': int})

    def test_getitem(self):
        self.assertIsType('range(x)[x-2]', int, env={'x': int})
        self.assertIsType('range(x)[2]', int, env={'x': int})
        self.assertIsType('range(x)[2:]', range, env={'x': int})

    def test_index(self):
        self.assertIsType('range(x).index(3)', int, env={'x': int})
        self.assertIsType('range(x).index(x, 2)', int, env={'x': int})
        self.assertIsType('range(x).index(x, 2, 6)', int, env={'x': int})
        self.assertIsType('range(x).index(1, 2, x - 6)', int, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('range(x).index(3, "2")', None, env={'x': int})

    def test_name(self):
        self.assertIsType('range.__name__', pentyping.Cst[range.__name__])

    def test_start(self):
        self.assertIsType('range(5).start', int)
        self.assertIsType('range(x).start', int, env={'x': int})

    def test_stop(self):
        self.assertIsType('range(5).stop', int)
        self.assertIsType('range(x).stop', int, env={'x': int})

    def test_step(self):
        self.assertIsType('range(5).step', int)
        self.assertIsType('range(x).step', int, env={'x': int})

    def test_str(self):
        self.assertIsType('str(range(x))', str, env={'x': int})

    def test_count(self):
        self.assertIsType('range(5).count(2)', int)
        self.assertIsType('range(x).count(2.)', int, env={'x': int})
        self.assertIsType('range(x).count("2")', int, env={'x': int})

    def test_eq(self):
        self.assertIsType('range(x) == range(2)', bool, env={'x': int})
        self.assertIsType('range(x) == 2', pentyping.Cst[False], env={'x': int})

    def test_ne(self):
        self.assertIsType('range(x) != range(2)', bool, env={'x': int})
        self.assertIsType('range(x) != 2', pentyping.Cst[True], env={'x': int})

    def test_ge(self):
        self.assertIsType('range(x) >= range(2)', bool, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('range(x) >= 2', None, env={'x': int})

    def test_gt(self):
        self.assertIsType('range(x) > range(2)', bool, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('range(x) > 2', None, env={'x': int})

    def test_le(self):
        self.assertIsType('range(x) <= range(2)', bool, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('range(x) <= 2', None, env={'x': int})

    def test_lt(self):
        self.assertIsType('range(x) < range(2)', bool, env={'x': int})
        with self.assertRaises(TypeError):
            self.assertIsType('range(x) < 2', None, env={'x': int})

    def test_iter(self):
        self.assertIsType('[e for e in range(x)]', pentyping.List[int],
                          env={'x': int})

    def test_reversed(self):
        self.assertIsType('[e for e in reversed(range(x))]', pentyping.List[int],
                          env={'x': int})

class TestStr(TestPenty):

    def test_iter(self):
        self.assertIsType('x.__iter__().__next__()', str, env={'x': str})
