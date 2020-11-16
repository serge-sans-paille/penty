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
    pass

def make_dtype_test(dtype):
    def test_dtype(self):
        self.assertIsType('abs(x)', dtype, env={'x': dtype})
        self.assertIsType('x + x', dtype, env={'x': dtype})
        self.assertIsType('x + 1', type(dtype() + 1), env={'x': dtype})
        self.assertIsType('x & x', dtype, env={'x': dtype})
        self.assertIsType('x & np.uint8(1)', type(dtype(1) & np.uint8(1)), env={'x': dtype})
        self.assertIsType('bool(x)', bool, env={'x': dtype})
        self.assertIsType('divmod(x, y)', pentyping.Tuple[dtype, dtype],
                          env={'x': dtype, 'y': dtype})
        dmty = divmod(dtype(1), 1)
        self.assertIsType('divmod(x, 1)',
                          pentyping.Tuple[type(dmty[0]), type(dmty[1])],
                          env={'x': dtype, 'y': dtype})
        self.assertIsType('x == x', bool, env={'x': dtype})
        self.assertIsType('x == 1', bool, env={'x': dtype})
        self.assertIsType('x // x', dtype, env={'x': dtype})
        self.assertIsType('x // np.int8(1)', type(dtype(1) // np.int8(1)), env={'x': dtype})
        self.assertIsType('x >= x', np.bool_, env={'x': dtype})
        self.assertIsType('x >= 1', np.bool_, env={'x': dtype})
        self.assertIsType('x > x', np.bool_, env={'x': dtype})
        self.assertIsType('x > 1', np.bool_, env={'x': dtype})
        self.assertIsType('int(x)', int, env={'x': dtype})
        self.assertIsType('~x', dtype, env={'x': dtype})
        self.assertIsType('x <= x', np.bool_, env={'x': dtype})
        self.assertIsType('x <= 1', np.bool_, env={'x': dtype})
        self.assertIsType('x < x', np.bool_, env={'x': dtype})
        self.assertIsType('x < 1', np.bool_, env={'x': dtype})
        self.assertIsType('x << x', dtype, env={'x': dtype})
        self.assertIsType('x << np.uint8(1)', type(dtype(1) << np.uint8(1)), env={'x': dtype})
        self.assertIsType('x % x', dtype, env={'x': dtype})
        self.assertIsType('x % np.uint8(1)', type(dtype(1) % np.uint8(1)), env={'x': dtype})
        self.assertIsType('x * x', dtype, env={'x': dtype})
        self.assertIsType('x * np.uint8(1)', type(dtype(1) * np.uint8(1)), env={'x': dtype})
        self.assertIsType('x != x', bool, env={'x': dtype})
        self.assertIsType('x != 1', bool, env={'x': dtype})
        self.assertIsType('-x', dtype, env={'x': dtype})
        self.assertIsType('x | x', dtype, env={'x': dtype})
        self.assertIsType('x | np.uint8(1)', type(dtype(1) | np.uint8(1)), env={'x': dtype})
        self.assertIsType('+x', dtype, env={'x': dtype})
        self.assertIsType('x ** x', dtype, env={'x': dtype})
        self.assertIsType('x ** np.uint8(1)', type(dtype(1) * np.uint8(1)), env={'x': dtype})
        self.assertIsType('str(x)', str, env={'x': dtype})
        self.assertIsType('dtype()', pentyping.Cst[dtype()],
                          env={'dtype': pentyping.Type[dtype]})
        self.assertIsType('dtype(3)', pentyping.Cst[dtype(3)],
                          env={'dtype': pentyping.Type[dtype]})
        self.assertIsType('dtype(x)', dtype,
                          env={'x': int, 'dtype': pentyping.Type[dtype]})
        self.assertIsType('dtype(x)', dtype,
                          env={'x': float, 'dtype': pentyping.Type[dtype]})
        self.assertIsType('dtype(x)', dtype,
                          env={'x': str, 'dtype': pentyping.Type[dtype]})
        self.assertIsType('x - x', dtype, env={'x': dtype})
        self.assertIsType('x - np.uint8(1)', type(dtype(1) - np.uint8(1)), env={'x': dtype})
        self.assertIsType('x / x', type(dtype(1)/dtype(1)), env={'x': dtype})
        self.assertIsType('x / np.uint8(1)', type(dtype(1) / np.uint8(1)), env={'x': dtype})
        self.assertIsType('x ^ x', dtype, env={'x': dtype})
        self.assertIsType('x ^ np.uint8(1)', type(dtype(1) ^ np.uint8(1)), env={'x': dtype})
        self.assertIsType('x.__round__()', int, env={'x': dtype})
        self.assertIsType('dtype(3).__round__()',
                          pentyping.Cst[dtype(3).__round__()],
                          env={'dtype': pentyping.Type[dtype]})
        self.assertIsType('x.denominator', pentyping.Cst[1], env={'x': dtype})
        self.assertIsType('x.numerator', dtype, env={'x': dtype})
        self.assertIsType('x.imag', pentyping.Cst[dtype(0)], env={'x': dtype})
        self.assertIsType('x.itemsize', pentyping.Cst[dtype(0).itemsize], env={'x': dtype})
        self.assertIsType('x.nbytes', pentyping.Cst[dtype(0).nbytes], env={'x': dtype})
        self.assertIsType('x.ndim', pentyping.Cst[0], env={'x': dtype})
        self.assertIsType('x.real', dtype, env={'x': dtype})
        self.assertIsType('x.shape', pentyping.Tuple[()], env={'x': dtype})
        self.assertIsType('x.size', pentyping.Cst[1], env={'x': dtype})
        self.assertIsType('x.strides', pentyping.Tuple[()], env={'x': dtype})
        self.assertIsType('1 + x', type(1 + dtype(1)), env={'x': dtype})
        self.assertIsType('1 // x', type(1 // dtype(1)), env={'x': dtype})
        self.assertIsType('1 % x', type(1 % dtype(1)), env={'x': dtype})
        self.assertIsType('1 * x', type(1 * dtype(1)), env={'x': dtype})
        self.assertIsType('1 ** x', type(1 ** dtype(1)), env={'x': dtype})
        self.assertIsType('1 - x', type(1 - dtype(1)), env={'x': dtype})
        self.assertIsType('1 / x', type(1 / dtype(1)), env={'x': dtype})

    return test_dtype


for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
              np.int64, np.uint64):
    setattr(TestDtype,
            'test_{}'.format(dtype.__name__),
            make_dtype_test(dtype))

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
