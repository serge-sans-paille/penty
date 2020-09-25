from unittest import TestCase
import gast as ast
import penty
import typing

class TestExpr(TestCase):

    def assertIsType(self, expr, ty, env={}):
        if not isinstance(ty, set):
            ty = {ty}
        env = {k: v if isinstance(v, set) else {v}
               for k, v in env.items()}
        self.assertEqual(penty.type_eval(expr, env), ty)

    def test_boolop_ty(self):
        self.assertIsType('x or y', int, env={'x':int, 'y':int})
        self.assertIsType('x or y', {bool, int}, env={'x':bool, 'y':int})
        self.assertIsType('x or y or z', {bool, int, str}, env={'x':bool,
                                                                'y':int,
                                                                'z':str})
        self.assertIsType('x and y', int, env={'x':int, 'y':int})
        self.assertIsType('x and y', {bool, int}, env={'x':bool, 'y':int})
        self.assertIsType('x and y and z', {bool, int, str}, env={'x':bool,
                                                                'y':int,
                                                                'z':str})

    def test_binaryop_ty(self):
        self.assertIsType('x + x', int, env={"x": int})
        self.assertIsType('x - x', int, env={"x": int})
        self.assertIsType('x * x', int, env={"x": int})
        #self.assertIsType('x @ x', int, env={"x": int})
        self.assertIsType('x / x', float, env={"x": int})
        self.assertIsType('x % x', int, env={"x": int})
        self.assertIsType('x // x', int, env={"x": int})
        self.assertIsType('x | x', int, env={"x": int})
        self.assertIsType('x & x', int, env={"x": int})
        self.assertIsType('x ^ x', int, env={"x": int})
        self.assertIsType('x ** x', int, env={"x": int})

    def test_unaryop_ty(self):
        self.assertIsType('+x', int, env={'x': int})
        self.assertIsType('-x', int, env={'x': int})
        self.assertIsType('~x', int, env={'x': int})
        self.assertIsType('not x', bool, env={'x': int})

    def test_lambda_ty(self):
        self.assertIsType('(lambda : x)()', int, env={'x': int})
        self.assertIsType('(lambda x: x)(x)', int, env={'x': int})
        self.assertIsType('(lambda x, y: x and y)(x, y)', {int, float},
                          env={'x': int, 'y': float})

    def test_ifexpr_ty(self):
        self.assertIsType('x if x else y', {int, float}, env={'x': int, 'y':
                                                              float})
        self.assertIsType('x if True else y', int, env={'x': int, 'y':
                                                        float})
        self.assertIsType('x if False else y', float, env={'x': int, 'y':
                                                           float})
        self.assertIsType('x if 1 else y', int, env={'x': int, 'y':
                                                        float})
        self.assertIsType('x if 0 else y', float, env={'x': int, 'y':
                                                           float})

    def test_dict_ty(self):
        self.assertIsType('{}', dict)
        self.assertIsType('{x:y}', typing.Dict[int, str], env={'x': int,
                                                                 'y': str})
        self.assertIsType('{x:y, y:x}',
                          {typing.Dict[int, str], typing.Dict[str, int]},
                          env={'x': int, 'y': str})
        self.assertIsType('{x:y, y:x}',
                          typing.Dict[int, int],
                          env={'x': int, 'y': int})

    def test_set_ty(self):
        self.assertIsType('{x}', typing.Set[int], env={'x': int})
        self.assertIsType('{x, x}', typing.Set[int], env={'x': int})
        self.assertIsType('{x, y}',
                          {typing.Set[float], typing.Set[int]},
                          env={'x': int, 'y': float})

    def test_constant_ty(self):
        self.assertIsType('True', True)
        self.assertIsType('False', False)
        self.assertIsType('None', None)
        self.assertIsType('1', 1)
        self.assertIsType('1.', 1.)
        self.assertIsType('1.j', 1.j)
        self.assertIsType('""', "")

    def test_contant_op_ty(self):
        self.assertIsType('+(1)', +1)
        self.assertIsType('-(1)', -1)
        self.assertIsType('~1', ~1)
        self.assertIsType('not 1', not 1)
        self.assertIsType('1 + 1', 1+1)
        self.assertIsType('1 - 1', 1 - 1)
        self.assertIsType('1 * 1', 1 * 1)
        #self.assertIsType('x @ x', int, env={"x": int})
        self.assertIsType('1 / 1', 1 / 1)
        self.assertIsType('1 % 1', 1 % 1)
        self.assertIsType('1 // 2', 1 // 2)
        self.assertIsType('1 | 2', 1 | 2)
        self.assertIsType('1 & 3', 1 & 3)
        self.assertIsType('2 ^ 3', 2 ^3)
        self.assertIsType('2 ** 3', 2 ** 3)
