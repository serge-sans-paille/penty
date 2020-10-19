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

    def test_len(self):
        self.assertIsType('len(x)',
                          int,
                          env={'x': NDArray[float, typing.Tuple[int, int]]})
        self.assertIsType('len(x)',
                          pentyping.Cst[1],
                          env={'x': NDArray[float, typing.Tuple[pentyping.Cst[1], int]]})

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
