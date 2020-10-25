from unittest import TestCase
import gast as ast
import penty
import penty.types as pentyping
from penty.pentypes.numpy import NDArray
import typing

class TestNumpy(TestCase):

    def assertIsType(self, expr, ty, env={}):
        if not isinstance(ty, set):
            ty = {ty}
        env = {k: v if isinstance(v, set) else {v}
               for k, v in env.items()}
        env = penty.type_exec("import numpy as np", env)
        self.assertEqual(penty.type_eval(expr, env), ty)

    def test_ones(self):
        self.assertIsType('np.ones(x)',
                          NDArray[float, typing.Tuple[int]],
                          env={'x':int})
        self.assertIsType('np.ones(1)',
                          NDArray[float, typing.Tuple[pentyping.Cst[1]]],
                          env={'x':int})
        self.assertIsType('np.ones((1, x))',
                          NDArray[float, typing.Tuple[pentyping.Cst[1], int]],
                          env={'x':int})
        self.assertIsType('np.ones(x, int)',
                          NDArray[int, typing.Tuple[int]],
                          env={'x':int})
        self.assertIsType('np.ones(1, int)',
                          NDArray[int, typing.Tuple[pentyping.Cst[1]]],
                          env={'x':int})
        self.assertIsType('np.ones((1, x), int)',
                          NDArray[int, typing.Tuple[pentyping.Cst[1], int]],
                          env={'x':int})

class TestNDArray(TestCase):

    def assertIsType(self, expr, ty, env={}):
        if not isinstance(ty, set):
            ty = {ty}
        env = {k: v if isinstance(v, set) else {v}
               for k, v in env.items()}
        env = penty.type_exec("import numpy as np", env)
        self.assertEqual(penty.type_eval(expr, env), ty)

    def test_abs(self):
        self.assertIsType('x.__abs__()',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x.__abs__()',
                          NDArray[int, typing.Tuple[pentyping.Cst[5]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5]]]})

    def test_getitem(self):
        self.assertIsType('x[0]',
                          NDArray[float, typing.Tuple[int]],
                          env={'x': NDArray[float, typing.Tuple[int, int]]})
        self.assertIsType('x[y]',
                          NDArray[float, typing.Tuple[pentyping.Cst[3]]],
                          env={'x': NDArray[float, typing.Tuple[int, pentyping.Cst[3]]],
                               'y': int})
        self.assertIsType('x[0, y]',
                          float,
                          env={'x': NDArray[float, typing.Tuple[int, int]],
                               'y': int})
        self.assertIsType('x[0:3]',
                          NDArray[float, typing.Tuple[pentyping.Cst[3], int]],
                          env={'x': NDArray[float, typing.Tuple[int, int]]})
        self.assertIsType('x[0:y]',
                          NDArray[float, typing.Tuple[int, int]],
                          env={'x': NDArray[float, typing.Tuple[int, int]],
                               'y': int})
        self.assertIsType('x[0:3, 1]',
                          NDArray[float, typing.Tuple[pentyping.Cst[3]]],
                          env={'x': NDArray[float, typing.Tuple[int, int]]})
        self.assertIsType('x[0:y, 1]',
                          NDArray[float, typing.Tuple[int]],
                          env={'x': NDArray[float, typing.Tuple[int, int]],
                               'y': int})
        self.assertIsType('x[0:3, :1]',
                          NDArray[float, typing.Tuple[pentyping.Cst[3],
                                                      pentyping.Cst[1]]],
                          env={'x': NDArray[float, typing.Tuple[int, int]]})
        self.assertIsType('x[0:y, :1]',
                          NDArray[float, typing.Tuple[int,
                                                      pentyping.Cst[1]]],
                          env={'x': NDArray[float, typing.Tuple[int, int]],
                               'y': int})

    def test_add(self):
        self.assertIsType('x + 1',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x + 1.',
                          NDArray[float, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x + x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x + x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x + y',
                          NDArray[float, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int,
                                                              pentyping.Cst[1]]],
                               'y': NDArray[float, typing.Tuple[int, pentyping.Cst[1]]]
                         })
        self.assertIsType('x + x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_and(self):
        self.assertIsType('x & 1',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x & x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x & x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x & x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})
        with self.assertRaises(TypeError):
            env = {'x': {NDArray[float, typing.Tuple[int, int]]}}
            penty.type_eval('x & 1', env)

    def test_bool(self):
        self.assertIsType('bool(x)',
                          bool,
                          env={'x': NDArray[float, typing.Tuple[int, int]]})
        self.assertIsType('bool(x)',
                          bool,
                          env={'x': NDArray[float, typing.Tuple[pentyping.Cst[1], int]]})
        self.assertIsType('bool(x)',
                          pentyping.Cst[True],
                          env={'x': NDArray[float,
                                            typing.Tuple[pentyping.Cst[1]]]})
        self.assertIsType('bool(x)',
                          pentyping.Cst[False],
                          env={'x': NDArray[float, typing.Tuple[pentyping.Cst[0]]]})


    def test_eq(self):
        self.assertIsType('x == 1',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x == x[1]',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x == x[1]',
                          NDArray[bool, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x == x[1]',
                          NDArray[bool, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_floordiv(self):
        self.assertIsType('x // 1',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x // x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x // x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x // x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_ge(self):
        self.assertIsType('x >= 1',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x >= x[1]',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x >= x[1]',
                          NDArray[bool, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x >= x[1]',
                          NDArray[bool, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_gt(self):
        self.assertIsType('x > 1',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x > x[1]',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x > x[1]',
                          NDArray[bool, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x > x[1]',
                          NDArray[bool, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_invert(self):
        with self.assertRaises(TypeError):
            env = {'x': {NDArray[float, typing.Tuple[int, int]]}}
            penty.type_eval('~x', env)

        self.assertIsType('~x',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})

    def test_le(self):
        self.assertIsType('x <= 1',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x <= x[1]',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x <= x[1]',
                          NDArray[bool, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x <= x[1]',
                          NDArray[bool, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_len(self):
        self.assertIsType('len(x)',
                          int,
                          env={'x': NDArray[float, typing.Tuple[int, int]]})
        self.assertIsType('len(x)',
                          pentyping.Cst[1],
                          env={'x': NDArray[float, typing.Tuple[pentyping.Cst[1], int]]})

    def test_lshift(self):
        self.assertIsType('x << 1',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x << x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x << x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x << x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_lt(self):
        self.assertIsType('x < 1',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x < x[1]',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x < x[1]',
                          NDArray[bool, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x < x[1]',
                          NDArray[bool, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_rshift(self):
        self.assertIsType('x >> 1',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x >> x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x >> x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x >> x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_matmul(self):
        with self.assertRaises(TypeError):
            env = {'x': {NDArray[int, typing.Tuple[int, int]]}}
            penty.type_eval('x @ 1', env)

        self.assertIsType('x @ x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x @ x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x @ x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_mod(self):
        self.assertIsType('x % 1',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x % x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x % x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x % x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_mul(self):
        self.assertIsType('x * 1',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x * x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x * x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x * x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_ne(self):
        self.assertIsType('x != 1',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x != x[1]',
                          NDArray[bool, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x != x[1]',
                          NDArray[bool, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x != x[1]',
                          NDArray[bool, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_neg(self):
        self.assertIsType('-x',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('-x',
                          NDArray[int, typing.Tuple[pentyping.Cst[5]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5]]]})

    def test_or(self):
        self.assertIsType('x | 1',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x | x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x | x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x | x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_pos(self):
        self.assertIsType('+x',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('+x',
                          NDArray[int, typing.Tuple[pentyping.Cst[5]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5]]]})

    def test_pow(self):
        self.assertIsType('x ** 1',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x ** x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x ** x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x ** x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_str(self):
        self.assertIsType('x.__str__()',
                          str,
                          env={'x': NDArray[int, typing.Tuple[int, int]]})


    def test_sub(self):
        self.assertIsType('x - 1',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x - x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x - x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x - x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_truediv(self):
        self.assertIsType('x / 1',
                          NDArray[float, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x / x[1]',
                          NDArray[float, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x / x[1]',
                          NDArray[float, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x / x[1]',
                          NDArray[float, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_xor(self):
        self.assertIsType('x ^ 1',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x ^ x[1]',
                          NDArray[int, typing.Tuple[int, int]],
                          env={'x': NDArray[int, typing.Tuple[int, int]]})
        self.assertIsType('x ^ x[1]',
                          NDArray[int, typing.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x ^ x[1]',
                          NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, typing.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})
