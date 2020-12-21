from unittest import TestCase
import gast as ast
import penty
import penty.types as pentyping
from penty.pentypes.numpy import NDArray, register as register_numpy
import typing
import numpy as np

from pentest import TestPenty, inject_spec_test

class TestNumpyBase(TestPenty):
    def assertIsType(self, expr, ty, env={}):
        env = penty.type_exec("import numpy as np", env.copy())
        super(TestNumpyBase, self).assertIsType(expr, ty, env)

class TestNumpySpecs(TestCase):
    pass

register_numpy(penty.penty.Types)
inject_spec_test(TestNumpySpecs, pentyping.Module['numpy'], np, 'numpy')


class TestDtype(TestNumpyBase):
    pass

def make_integer_dtype_test(dtype):
    def test_dtype(self):
        self.assertIsType('abs(x)', dtype, env={'x': dtype})
        self.assertIsType('bin(x)', str, env={'x': dtype})
        self.assertIsType('hex(x)', str, env={'x': dtype})
        self.assertIsType('oct(x)', str, env={'x': dtype})
        self.assertIsType('round(x, 2)', type(round(dtype(1), 2)), env={'x': dtype})
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
            make_integer_dtype_test(dtype))

def make_floating_dtype_test(dtype):
    def test_dtype(self):
        self.assertIsType('abs(x)', dtype, env={'x': dtype})
        self.assertIsType('x + x', dtype, env={'x': dtype})
        self.assertIsType('x + 1', type(dtype() + 1), env={'x': dtype})
        self.assertIsType('bool(x)', bool, env={'x': dtype})
        self.assertIsType('divmod(x, y)', pentyping.Tuple[dtype, dtype],
                          env={'x': dtype, 'y': dtype})
        dmty = divmod(dtype(1), 1)
        self.assertIsType('divmod(x, 1)',
                          pentyping.Tuple[type(dmty[0]), type(dmty[1])],
                          env={'x': dtype, 'y': dtype})
        self.assertIsType('round(x, 2)', dtype, env={'x': dtype})
        self.assertIsType('x == x', bool, env={'x': dtype})
        self.assertIsType('x == 1', bool, env={'x': dtype})
        self.assertIsType('x // x', dtype, env={'x': dtype})
        self.assertIsType('x // np.int8(1)', type(dtype(1) // np.int8(1)), env={'x': dtype})
        self.assertIsType('x >= x', np.bool_, env={'x': dtype})
        self.assertIsType('x >= 1', np.bool_, env={'x': dtype})
        self.assertIsType('x > x', np.bool_, env={'x': dtype})
        self.assertIsType('x > 1', np.bool_, env={'x': dtype})
        self.assertIsType('int(x)', int, env={'x': dtype})
        self.assertIsType('float(x)', float, env={'x': dtype})
        self.assertIsType('x <= x', np.bool_, env={'x': dtype})
        self.assertIsType('x <= 1', np.bool_, env={'x': dtype})
        self.assertIsType('x < x', np.bool_, env={'x': dtype})
        self.assertIsType('x < 1', np.bool_, env={'x': dtype})
        self.assertIsType('x % x', dtype, env={'x': dtype})
        self.assertIsType('x % np.uint8(1)', type(dtype(1) % np.uint8(1)), env={'x': dtype})
        self.assertIsType('x * x', dtype, env={'x': dtype})
        self.assertIsType('x * np.uint8(1)', type(dtype(1) * np.uint8(1)), env={'x': dtype})
        self.assertIsType('x != x', bool, env={'x': dtype})
        self.assertIsType('x != 1', bool, env={'x': dtype})
        self.assertIsType('-x', dtype, env={'x': dtype})
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
        self.assertIsType('x.__round__()', int, env={'x': dtype})
        self.assertIsType('dtype(3).__round__()',
                          pentyping.Cst[dtype(3).__round__()],
                          env={'dtype': pentyping.Type[dtype]})
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
        air = dtype(1).as_integer_ratio()
        self.assertIsType('x.as_integer_ratio()',
                          pentyping.Tuple[type(air[0]), type(air[1])],
                          env={'x': dtype})
        self.assertIsType('dtype(1).as_integer_ratio()',
                          pentyping.Tuple[pentyping.Cst[1], pentyping.Cst[1]],
                          env={'dtype': pentyping.Type[dtype]})

    return test_dtype


for dtype in (np.float32, np.float64):
    setattr(TestDtype,
            'test_{}'.format(dtype.__name__),
            make_floating_dtype_test(dtype))

def make_complex_dtype_test(dtype):
    def test_dtype(self):
        self.assertIsType('abs(x)', dtype, env={'x': dtype})
        self.assertIsType('x + x', dtype, env={'x': dtype})
        self.assertIsType('x + 1', type(dtype() + 1), env={'x': dtype})
        self.assertIsType('bool(x)', bool, env={'x': dtype})
        self.assertIsType('x == x', bool, env={'x': dtype})
        self.assertIsType('x == 1', bool, env={'x': dtype})
        self.assertIsType('x // x', dtype, env={'x': dtype})
        self.assertIsType('x // np.int8(1)', type(dtype(1) // np.int8(1)), env={'x': dtype})
        self.assertIsType('x >= x', np.bool_, env={'x': dtype})
        self.assertIsType('x >= 1', np.bool_, env={'x': dtype})
        self.assertIsType('x > x', np.bool_, env={'x': dtype})
        self.assertIsType('x > 1', np.bool_, env={'x': dtype})
        self.assertIsType('int(x)', int, env={'x': dtype})
        self.assertIsType('float(x)', float, env={'x': dtype})
        self.assertIsType('complex(x)', complex, env={'x': dtype})
        self.assertIsType('x <= x', np.bool_, env={'x': dtype})
        self.assertIsType('x <= 1', np.bool_, env={'x': dtype})
        self.assertIsType('x < x', np.bool_, env={'x': dtype})
        self.assertIsType('x < 1', np.bool_, env={'x': dtype})
        self.assertIsType('x * x', dtype, env={'x': dtype})
        self.assertIsType('x * np.uint8(1)', type(dtype(1) * np.uint8(1)), env={'x': dtype})
        self.assertIsType('x != x', bool, env={'x': dtype})
        self.assertIsType('x != 1', bool, env={'x': dtype})
        self.assertIsType('-x', dtype, env={'x': dtype})
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
        self.assertIsType('1 * x', type(1 * dtype(1)), env={'x': dtype})
        self.assertIsType('1 ** x', type(1 ** dtype(1)), env={'x': dtype})
        self.assertIsType('1 - x', type(1 - dtype(1)), env={'x': dtype})
        self.assertIsType('1 / x', type(1 / dtype(1)), env={'x': dtype})

    return test_dtype


for dtype in (np.complex64, np.complex128):
    setattr(TestDtype,
            'test_{}'.format(dtype.__name__),
            make_complex_dtype_test(dtype))

class TestNumpy(TestNumpyBase):

    def make_test_oez(self, oez):
        self.assertIsType('oez(x)',
                          NDArray[float, pentyping.Tuple[int]],
                          env={'x':int, 'oez': oez})
        self.assertIsType('oez(1)',
                          NDArray[float, pentyping.Tuple[pentyping.Cst[1]]],
                          env={'x':int, 'oez': oez})
        self.assertIsType('oez((1, x))',
                          NDArray[float, pentyping.Tuple[pentyping.Cst[1], int]],
                          env={'x':int, 'oez': oez})
        self.assertIsType('oez(x, int)',
                          NDArray[int, pentyping.Tuple[int]],
                          env={'x':int, 'oez': oez})
        self.assertIsType('oez(1, int)',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[1]]],
                          env={'x':int, 'oez': oez})
        self.assertIsType('oez((1, x), int)',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[1], int]],
                          env={'x':int, 'oez': oez})

    def test_ones(self):
        ones_ty = penty.penty.Types[pentyping.Module['numpy']]['ones']
        self.make_test_oez(ones_ty)

    def test_empty(self):
        empty_ty = penty.penty.Types[pentyping.Module['numpy']]['empty']
        self.make_test_oez(empty_ty)

    def test_zeros(self):
        zeros_ty = penty.penty.Types[pentyping.Module['numpy']]['zeros']
        self.make_test_oez(zeros_ty)

    def make_test_oez_like(self, oez_like):
        self.assertIsType('oez_like(x)',
                          NDArray[float, pentyping.Tuple[int]],
                          env={'x':NDArray[float, pentyping.Tuple[int]],
                               'oez_like': oez_like})
        self.assertIsType('oez_like(x)',
                          NDArray[complex, pentyping.Tuple[pentyping.Cst[3]]],
                          env={'x':NDArray[complex,
                                           pentyping.Tuple[pentyping.Cst[3]]],
                               'oez_like': oez_like})
        self.assertIsType('oez_like(x, dtype=None, order="K", subok=True, shape=3)',
                          NDArray[float, pentyping.Tuple[pentyping.Cst[3]]],
                          env={'x':NDArray[float, pentyping.Tuple[int]],
                               'oez_like': oez_like})
        self.assertIsType('oez_like(x, dtype=int, order="K", subok=False, shape=None)',
                          NDArray[int, pentyping.Tuple[int]],
                          env={'x':NDArray[float, pentyping.Tuple[int]],
                               'oez_like': oez_like})

    def test_ones_like(self):
        ones_like_ty = penty.penty.Types[pentyping.Module['numpy']]['ones_like']
        self.make_test_oez_like(ones_like_ty)

    def test_empty_like(self):
        empty_like_ty = penty.penty.Types[pentyping.Module['numpy']]['empty_like']
        self.make_test_oez_like(empty_like_ty)

    def test_zeros_like(self):
        zeros_like_ty = penty.penty.Types[pentyping.Module['numpy']]['zeros_like']
        self.make_test_oez_like(zeros_like_ty)


class TestNDArray(TestNumpyBase):

    def test_abs(self):
        self.assertIsType('x.__abs__()',
                          NDArray[int, pentyping.Tuple[int, int]],
                          env={'x': NDArray[int, pentyping.Tuple[int, int]]})
        self.assertIsType('x.__abs__()',
                          NDArray[int, pentyping.Tuple[pentyping.Cst[5]]],
                          env={'x': NDArray[int, pentyping.Tuple[pentyping.Cst[5]]]})

    def test_hash(self):
        self.assertIsType('x.__hash__', pentyping.Cst[None],
                          env={'x': NDArray[int, pentyping.Tuple[int]]})

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
