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
        self.assertIsType('x if x else y',
                          {int, float},
                          env={'x': int, 'y': float})
        self.assertIsType('x if True else y',
                          int, env={'x': int, 'y': float})
        self.assertIsType('x if False else y',
                          float, env={'x': int, 'y': float})
        self.assertIsType('x if 1 else y',
                          int, env={'x': int, 'y': float})
        self.assertIsType('x if 0 else y',
                          float, env={'x': int, 'y': float})

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

    def test_list_comp_ty(self):
        self.assertIsType('[x for x in y]', typing.List[str], env={'y': str})
        self.assertIsType('[x for x in y if 0]', list, env={'y': str})
        self.assertIsType('[1 for x in y if y]', typing.List[int], env={'y': str})
        self.assertIsType('[1 for x in y for z in y]', typing.List[int], env={'y': str})

    def test_set_comp_ty(self):
        self.assertIsType('{x for x in y}', typing.Set[str], env={'y': str})
        self.assertIsType('{x for x in y if 0}', set, env={'y': str})
        self.assertIsType('{1 for x in y if y}', typing.Set[int], env={'y': str})
        self.assertIsType('{1 for x in y for z in y}', typing.Set[int], env={'y': str})

    def test_dict_comp_ty(self):
        self.assertIsType('{x:1 for x in y}', typing.Dict[str, int], env={'y': str})
        self.assertIsType('{1:x for x in y if 0}', dict, env={'y': str})
        self.assertIsType('{1:1 for x in y if y}', typing.Dict[int, int], env={'y': str})
        self.assertIsType('{1:"" for x in y for z in y}', typing.Dict[int,str], env={'y': str})

    def test_gen_expr_ty(self):
        self.assertIsType('(x for x in y)',
                          typing.Generator[str, None, None],
                          env={'y': str})
        self.assertIsType('(x for x in y if 0)',
                          typing.Generator,
                          env={'y': str})
        self.assertIsType('(1 for x in y if y)',
                          typing.Generator[int, None, None],
                          env={'y': str})
        self.assertIsType('(1 for x in y for z in y)',
                          typing.Generator[int, None, None],
                          env={'y': str})

    def test_compare_ty(self):
        self.assertIsType('x == y == x', bool, env={'x': int, 'y': int})
        self.assertIsType('0 == 1 == x', penty.types.Cst[False], env={'x': int, 'y': int})
        self.assertIsType('0 == 0 == x', bool, env={'x': int, 'y': int})
        self.assertIsType('0 == 0 == 0', penty.types.Cst[True], env={'x': int, 'y': int})

        self.assertIsType('x < y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 < y', bool, env={'x': int, 'y': int})
        self.assertIsType('x < 1', bool, env={'x': int, 'y': int})
        self.assertIsType('0 < 1', penty.types.Cst[True], env={'x': int, 'y': int})
        self.assertIsType('2 < 1', penty.types.Cst[False], env={'x': int, 'y': int})

        self.assertIsType('x <= y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 <= y', bool, env={'x': int, 'y': int})
        self.assertIsType('x <= 1', bool, env={'x': int, 'y': int})
        self.assertIsType('1 <= 1', penty.types.Cst[True], env={'x': int, 'y': int})
        self.assertIsType('2 <= 1', penty.types.Cst[False], env={'x': int, 'y': int})

        self.assertIsType('x == y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 == y', bool, env={'x': int, 'y': int})
        self.assertIsType('x == 1', bool, env={'x': int, 'y': int})
        self.assertIsType('1 == 1', penty.types.Cst[True], env={'x': int, 'y': int})
        self.assertIsType('1 == 2', penty.types.Cst[False], env={'x': int, 'y': int})

        self.assertIsType('x != y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 != y', bool, env={'x': int, 'y': int})
        self.assertIsType('x != 1', bool, env={'x': int, 'y': int})
        self.assertIsType('1 != 1', penty.types.Cst[False], env={'x': int, 'y': int})
        self.assertIsType('1 != 2', penty.types.Cst[True], env={'x': int, 'y': int})

        self.assertIsType('x > y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 > y', bool, env={'x': int, 'y': int})
        self.assertIsType('x > 1', bool, env={'x': int, 'y': int})
        self.assertIsType('1 > 1', penty.types.Cst[False], env={'x': int, 'y': int})
        self.assertIsType('1 > 0', penty.types.Cst[True], env={'x': int, 'y': int})

        self.assertIsType('x >= y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 >= y', bool, env={'x': int, 'y': int})
        self.assertIsType('x >= 1', bool, env={'x': int, 'y': int})
        self.assertIsType('0 >= 1', penty.types.Cst[False], env={'x': int, 'y': int})
        self.assertIsType('1 >= 0', penty.types.Cst[True], env={'x': int, 'y': int})

        self.assertIsType('x is y', bool, env={'x': int, 'y': int})
        self.assertIsType('x is y', penty.types.Cst[False], env={'x': int, 'y': float})
        self.assertIsType('1. is y', bool, env={'x': int, 'y': float})
        self.assertIsType('y is 1.', bool, env={'x': int, 'y': float})
        self.assertIsType('1 is y', penty.types.Cst[False], env={'x': int, 'y': float})
        self.assertIsType('y is 1', penty.types.Cst[False], env={'x': int, 'y': float})

        self.assertIsType('x is not y', bool, env={'x': int, 'y': int})
        self.assertIsType('x is not y', penty.types.Cst[True], env={'x': int, 'y': float})

        #self.assertIsType('x in y', bool, env={'x': int, 'y': int})
        #self.assertIsType('x not in y', bool, env={'x': int, 'y': int})

    def test_repr(self):
        self.assertIsType('repr(x)', str, env={'x': int})
        self.assertIsType('repr(1)', penty.types.Cst['1'], env={'x': int})

    def test_constant_ty(self):
        self.assertIsType('True', penty.types.Cst[True])
        self.assertIsType('False', penty.types.Cst[False])
        self.assertIsType('None', penty.types.Cst[None])
        self.assertIsType('1', penty.types.Cst[1])
        self.assertIsType('1.', penty.types.Cst[1.])
        self.assertIsType('1.j', penty.types.Cst[1.j])
        self.assertIsType('""', penty.types.Cst[""])

    def test_contant_op_ty(self):
        self.assertIsType('+(1)', penty.types.Cst[+1])
        self.assertIsType('-(1)', penty.types.Cst[-1])
        self.assertIsType('~1', penty.types.Cst[~1])
        self.assertIsType('not 1', penty.types.Cst[not 1])
        self.assertIsType('1 + 1', penty.types.Cst[1+1])
        self.assertIsType('1 - 1', penty.types.Cst[1 - 1])
        self.assertIsType('1 * 1', penty.types.Cst[1 * 1])
        #self.assertIsType('x @ x', int, env={"x": int})
        self.assertIsType('1 / 1', penty.types.Cst[1 / 1])
        self.assertIsType('1 % 1', penty.types.Cst[1 % 1])
        self.assertIsType('1 // 2', penty.types.Cst[1 // 2])
        self.assertIsType('1 | 2', penty.types.Cst[1 | 2])
        self.assertIsType('1 & 3', penty.types.Cst[1 & 3])
        self.assertIsType('2 ^ 3', penty.types.Cst[2 ^3])
        self.assertIsType('2 ** 3', penty.types.Cst[2 ** 3])

    def test_subscript_ty(self):
        self.assertIsType('x[y]', int, env={'x': typing.List[int], 'y': int})
        self.assertIsType('x[0]', int, env={'x': typing.List[int]})
        self.assertIsType('x[y]', typing.List[int],
                          env={'x': typing.List[int], 'y': slice})
        self.assertIsType('x[0:y]', typing.List[int],
                          env={'x': typing.List[int], 'y': int})
        self.assertIsType('x[0:1]', typing.List[int],
                          env={'x': typing.List[int]})
        self.assertIsType('x[::]', typing.List[int],
                          env={'x': typing.List[int]})

        self.assertIsType('x[y]', {int, float},
                          env={'x': typing.Tuple[int, float], 'y': int})
        self.assertIsType('x[0]', int, env={'x': typing.Tuple[int, float]})
        self.assertIsType('x[y]', tuple,
                          env={'x': typing.Tuple[int, float], 'y': slice})
        self.assertIsType('x[0:y]', tuple,
                          env={'x': typing.Tuple[int, float], 'y': int})
        self.assertIsType('x[0:1]', typing.Tuple[int],
                          env={'x': typing.Tuple[int, float]})
        self.assertIsType('x[::]', typing.Tuple[int, float],
                          env={'x': typing.Tuple[int, float]})

    def test_list_ty(self):
        self.assertIsType('[]', list)
        self.assertIsType('[1]', typing.List[int])
        self.assertIsType('[x]', typing.List[int], env={'x': int})
        self.assertIsType('[1, 1.]', {typing.List[int], typing.List[float]})

    def test_tuple_ty(self):
        self.assertIsType('()', tuple)
        self.assertIsType('(1,)', (penty.types.Cst[1],))
        self.assertIsType('(1, x)', (penty.types.Cst[1], int), env={'x': int})
        self.assertIsType('(x, y)', (int, float), env={'x': int, 'y': float})
        self.assertIsType('(1, x or y)',
                          {(penty.types.Cst[1], float), (penty.types.Cst[1], int)},
                          env={'x': int, 'y': float})
