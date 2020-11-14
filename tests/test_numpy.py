from unittest import TestCase
import gast as ast
import penty
import penty.types as pentyping
from penty.pentypes.numpy import NDArray
import typing
import numpy as np

class TestPenty(TestCase):

    def assertTypesEqual(self, t0, t1):
        st0 = {str(ty) for ty in t0}
        st1 = {str(ty) for ty in t1}
        self.assertEqual(st0, st1)

    def assertIsType(self, expr, ty, env={}):
        if not isinstance(ty, set):
            ty = {ty}
        env = {k: v if isinstance(v, set) else {v}
               for k, v in env.items()}
        env = penty.type_exec("import numpy as np", env)
        self.assertTypesEqual(penty.type_eval(expr, env), ty)

class TestDtype(TestPenty):

    def test_int8(self):
        self.assertIsType('abs(x)', np.int8, env={'x': np.int8})
        self.assertIsType('x + x', np.int8, env={'x': np.int8})
        self.assertIsType('x + 1', np.int_, env={'x': np.int8})
        self.assertIsType('x & x', np.int8, env={'x': np.int8})
        self.assertIsType('x & 1', np.int_, env={'x': np.int8})
        self.assertIsType('bool(x)', bool, env={'x': np.int8})
        self.assertIsType('divmod(x, y)', pentyping.Tuple[np.int8, np.int8],
                          env={'x': np.int8, 'y': np.int8})
        self.assertIsType('divmod(x, 1)', pentyping.Tuple[np.int_, np.int_],
                          env={'x': np.int8, 'y': np.int8})
        self.assertIsType('x == x', bool, env={'x': np.int8})
        self.assertIsType('x == 1', bool, env={'x': np.int8})
        self.assertIsType('x // x', np.int8, env={'x': np.int8})
        self.assertIsType('x // 1', np.int_, env={'x': np.int8})
        self.assertIsType('x >= x', np.bool_, env={'x': np.int8})
        self.assertIsType('x >= 1', np.bool_, env={'x': np.int8})
        self.assertIsType('x > x', np.bool_, env={'x': np.int8})
        self.assertIsType('x > 1', np.bool_, env={'x': np.int8})
        self.assertIsType('int(x)', int, env={'x': np.int8})
        self.assertIsType('~x', np.int8, env={'x': np.int8})
        self.assertIsType('x <= x', np.bool_, env={'x': np.int8})
        self.assertIsType('x <= 1', np.bool_, env={'x': np.int8})
        self.assertIsType('x < x', np.bool_, env={'x': np.int8})
        self.assertIsType('x < 1', np.bool_, env={'x': np.int8})
        self.assertIsType('x << x', np.int8, env={'x': np.int8})
        self.assertIsType('x << 1', np.int_, env={'x': np.int8})
        self.assertIsType('x % x', np.int8, env={'x': np.int8})
        self.assertIsType('x % 1', np.int_, env={'x': np.int8})
        self.assertIsType('x * x', np.int8, env={'x': np.int8})
        self.assertIsType('x * 1', np.int_, env={'x': np.int8})
        self.assertIsType('x != x', bool, env={'x': np.int8})
        self.assertIsType('x != 1', bool, env={'x': np.int8})
        self.assertIsType('-x', np.int8, env={'x': np.int8})
        self.assertIsType('x | x', np.int8, env={'x': np.int8})
        self.assertIsType('x | 1', np.int_, env={'x': np.int8})
        self.assertIsType('+x', np.int8, env={'x': np.int8})
        self.assertIsType('x ** x', np.int8, env={'x': np.int8})
        self.assertIsType('x ** 1', np.int_, env={'x': np.int8})
        self.assertIsType('str(x)', str, env={'x': np.int8})
        self.assertIsType('np.int8()', pentyping.Cst[np.int8()])
        self.assertIsType('np.int8(3)', pentyping.Cst[np.int8(3)])
        self.assertIsType('np.int8(x)', np.int8, env={'x': int})
        self.assertIsType('np.int8(x)', np.int8, env={'x': float})
        self.assertIsType('np.int8(x)', np.int8, env={'x': str})
        self.assertIsType('x - x', np.int8, env={'x': np.int8})
        self.assertIsType('x - 1', np.int_, env={'x': np.int8})
        self.assertIsType('x / x', np.float32, env={'x': np.int8})
        self.assertIsType('x / 1', np.float64, env={'x': np.int8})
        self.assertIsType('x ^ x', np.int8, env={'x': np.int8})
        self.assertIsType('x ^ 1', np.int_, env={'x': np.int8})
        self.assertIsType('x.__round__()', int, env={'x': np.int8})
        self.assertIsType('np.int8(3).__round__()',
                          pentyping.Cst[np.int8(3).__round__()])
        self.assertIsType('x.denominator', pentyping.Cst[1], env={'x': np.int8})
        self.assertIsType('x.numerator', np.int8, env={'x': np.int8})
        self.assertIsType('x.imag', pentyping.Cst[np.int8(0)], env={'x': np.int8})
        self.assertIsType('x.itemsize', pentyping.Cst[np.int8(0).itemsize], env={'x': np.int8})
        self.assertIsType('x.nbytes', pentyping.Cst[np.int8(0).nbytes], env={'x': np.int8})
        self.assertIsType('x.ndim', pentyping.Cst[0], env={'x': np.int8})
        self.assertIsType('x.real', np.int8, env={'x': np.int8})
        self.assertIsType('x.shape', pentyping.Tuple[()], env={'x': np.int8})
        self.assertIsType('x.size', pentyping.Cst[1], env={'x': np.int8})
        self.assertIsType('x.strides', pentyping.Tuple[()], env={'x': np.int8})
        self.assertIsType('1 + x', np.int_, env={'x': np.int8})
        self.assertIsType('1 & x', np.int_, env={'x': np.int8})
        self.assertIsType('divmod(1, x)', pentyping.Tuple[np.int_, np.int_],
                          env={'x': np.int8})
        self.assertIsType('1 // x', np.int_, env={'x': np.int8})
        self.assertIsType('1 << x', np.int_, env={'x': np.int8})
        self.assertIsType('1 % x', np.int_, env={'x': np.int8})
        self.assertIsType('1 * x', np.int_, env={'x': np.int8})
        self.assertIsType('1 | x', np.int_, env={'x': np.int8})
        self.assertIsType('1 ** x', np.int_, env={'x': np.int8})
        self.assertIsType('1 >> x', np.int_, env={'x': np.int8})
        self.assertIsType('1 - x', np.int_, env={'x': np.int8})
        self.assertIsType('1 / x', np.float64, env={'x': np.int8})
        self.assertIsType('1 ^ x', np.int_, env={'x': np.int8})

class TestNumpy(TestPenty):

    def test_ones(self):
        self.assertIsType('np.ones(x)',
                          NDArray[float, pentyping.Tuple[int]],
                          env={'x':int})
        self.assertIsType('np.ones(1)',
                          NDArray[float, pentyping.Tuple[pentyping.Cst[1]]],
                          env={'x':int})
        self.assertIsType('np.ones((1, x))',
                          NDArray[float, pentyping.Tuple[pentyping.Cst[1], int]],
                          env={'x':int})
        self.assertIsType('np.ones(x, int)',
                          NDArray[int, pentyping.Tuple[int]],
                          env={'x':int})
        self.assertIsType('np.ones(1, int)',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[1]]],
                          env={'x':int})
        self.assertIsType('np.ones((1, x), int)',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[1], int]],
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
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x.__abs__()',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5]]]})

    def test_getitem(self):
        self.assertIsType('x[0]',
                          NDArray[float, pentyping.Tuple[int]],
                          env={'x': NDArray[float, pentyping.Tuple[int, int]]})
        self.assertIsType('x[y]',
                          NDArray[float, pentyping.Tuple[pentyping.Cst[3]]],
                          env={'x': NDArray[float, pentyping.Tuple[int, pentyping.Cst[3]]],
                               'y': int})
        self.assertIsType('x[0, y]',
                          float,
                          env={'x': NDArray[float, pentyping.Tuple[int, int]],
                               'y': int})
        self.assertIsType('x[0:3]',
                          NDArray[float, pentyping.Tuple[pentyping.Cst[3], int]],
                          env={'x': NDArray[float, pentyping.Tuple[int, int]]})
        self.assertIsType('x[0:y]',
                          NDArray[float, pentyping.Tuple[int, int]],
                          env={'x': NDArray[float, pentyping.Tuple[int, int]],
                               'y': int})
        self.assertIsType('x[0:3, 1]',
                          NDArray[float, pentyping.Tuple[pentyping.Cst[3]]],
                          env={'x': NDArray[float, pentyping.Tuple[int, int]]})
        self.assertIsType('x[0:y, 1]',
                          NDArray[float, pentyping.Tuple[int]],
                          env={'x': NDArray[float, pentyping.Tuple[int, int]],
                               'y': int})
        self.assertIsType('x[0:3, :1]',
                          NDArray[float, pentyping.Tuple[pentyping.Cst[3],
                                                      pentyping.Cst[1]]],
                          env={'x': NDArray[float, pentyping.Tuple[int, int]]})
        self.assertIsType('x[0:y, :1]',
                          NDArray[float, pentyping.Tuple[int,
                                                      pentyping.Cst[1]]],
                          env={'x': NDArray[float, pentyping.Tuple[int, int]],
                               'y': int})

    def test_add(self):
        self.assertIsType('x + 1',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x + 1.',
                          NDArray[float, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x + x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x + x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x + y',
                          NDArray[float, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int,
                                                              pentyping.Cst[1]]],
                               'y': NDArray[float, pentyping.Tuple[int, pentyping.Cst[1]]]
                         })
        self.assertIsType('x + x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_and(self):
        self.assertIsType('x & 1',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x & x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x & x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x & x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})
        with self.assertRaises(TypeError):
            env = {'x': {NDArray[float, pentyping.Tuple[int, int]]}}
            penty.type_eval('x & 1', env)

    def test_bool(self):
        self.assertIsType('bool(x)',
                          bool,
                          env={'x': NDArray[float, pentyping.Tuple[int, int]]})
        self.assertIsType('bool(x)',
                          bool,
                          env={'x': NDArray[float, pentyping.Tuple[pentyping.Cst[1], int]]})
        self.assertIsType('bool(x)',
                          pentyping.Cst[True],
                          env={'x': NDArray[float,
                                            pentyping.Tuple[pentyping.Cst[1]]]})
        self.assertIsType('bool(x)',
                          pentyping.Cst[False],
                          env={'x': NDArray[float, pentyping.Tuple[pentyping.Cst[0]]]})

    def test_dtype(self):
        self.assertIsType('x.dtype(0)',
                          pentyping.Cst[0.],
                          env={'x': NDArray[float, pentyping.Tuple[int]]})
        self.assertIsType('x.dtype is float',
                          pentyping.Cst[True],
                          env={'x': NDArray[float, pentyping.Tuple[pentyping.Cst[1]]]})

    def test_ndim(self):
        self.assertIsType('x.ndim',
                          pentyping.Cst[1],
                          env={'x': NDArray[float, pentyping.Tuple[int]]})
        self.assertIsType('x.ndim',
                          pentyping.Cst[2],
                          env={'x': NDArray[float, pentyping.Tuple[int, int]]})

    def test_shape(self):
        self.assertIsType('x.shape',
                          pentyping.Tuple[int],
                          env={'x': NDArray[float, pentyping.Tuple[int]]})
        self.assertIsType('x.shape',
                          pentyping.Tuple[pentyping.Cst[1]],
                          env={'x': NDArray[float, pentyping.Tuple[pentyping.Cst[1]]]})

    def test_eq(self):
        self.assertIsType('x == 1',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x == x[1]',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x == x[1]',
                          NDArray[bool, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x == x[1]',
                          NDArray[bool, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_floordiv(self):
        self.assertIsType('x // 1',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x // x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x // x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x // x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_ge(self):
        self.assertIsType('x >= 1',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x >= x[1]',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x >= x[1]',
                          NDArray[bool, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x >= x[1]',
                          NDArray[bool, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_gt(self):
        self.assertIsType('x > 1',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x > x[1]',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x > x[1]',
                          NDArray[bool, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x > x[1]',
                          NDArray[bool, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_invert(self):
        with self.assertRaises(TypeError):
            env = {'x': {NDArray[float, pentyping.Tuple[int, int]]}}
            penty.type_eval('~x', env)

        self.assertIsType('~x',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})

    def test_le(self):
        self.assertIsType('x <= 1',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x <= x[1]',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x <= x[1]',
                          NDArray[bool, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x <= x[1]',
                          NDArray[bool, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_len(self):
        self.assertIsType('len(x)',
                          int,
                          env={'x': NDArray[float, pentyping.Tuple[int, int]]})
        self.assertIsType('len(x)',
                          pentyping.Cst[1],
                          env={'x': NDArray[float, pentyping.Tuple[pentyping.Cst[1], int]]})

    def test_lshift(self):
        self.assertIsType('x << 1',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x << x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x << x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x << x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_lt(self):
        self.assertIsType('x < 1',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x < x[1]',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x < x[1]',
                          NDArray[bool, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x < x[1]',
                          NDArray[bool, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_rshift(self):
        self.assertIsType('x >> 1',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x >> x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x >> x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x >> x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_matmul(self):
        with self.assertRaises(TypeError):
            env = {'x': {NDArray[int, pentyping.Tuple[int, int]]}}
            penty.type_eval('x @ 1', env)

        self.assertIsType('x @ x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x @ x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x @ x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_mod(self):
        self.assertIsType('x % 1',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x % x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x % x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x % x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_mul(self):
        self.assertIsType('x * 1',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x * x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x * x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x * x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_ne(self):
        self.assertIsType('x != 1',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x != x[1]',
                          NDArray[bool, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x != x[1]',
                          NDArray[bool, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x != x[1]',
                          NDArray[bool, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_neg(self):
        self.assertIsType('-x',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('-x',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5]]]})

    def test_or(self):
        self.assertIsType('x | 1',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x | x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x | x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x | x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_pos(self):
        self.assertIsType('+x',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('+x',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5]]]})

    def test_pow(self):
        self.assertIsType('x ** 1',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x ** x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x ** x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x ** x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_str(self):
        self.assertIsType('x.__str__()',
                          str,
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})


    def test_sub(self):
        self.assertIsType('x - 1',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x - x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x - x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x - x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_truediv(self):
        self.assertIsType('x / 1',
                          NDArray[float, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x / x[1]',
                          NDArray[float, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x / x[1]',
                          NDArray[float, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x / x[1]',
                          NDArray[float, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})

    def test_xor(self):
        self.assertIsType('x ^ 1',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x ^ x[1]',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x ^ x[1]',
                          NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[int, pentyping.Cst[1]]]})
        self.assertIsType('x ^ x[1]',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5], pentyping.Cst[1]]]})
