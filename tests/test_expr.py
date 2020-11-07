from unittest import TestCase
import gast as ast
import penty
import penty.types as pentyping
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
        self.assertIsType('x << x', int, env={"x": int})
        self.assertIsType('x >> x', int, env={"x": int})
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
        self.assertIsType('y if x is None else abs(x)',
                          float,
                          env={'x': {pentyping.Cst[None], float},
                               'y': float})
        self.assertIsType('x if x is None else (x.append(1), x)[1]',
                          {pentyping.Cst[None], pentyping.List[int]},
                          env={'x': {pentyping.Cst[None], list}})

    def test_dict_ty(self):
        self.assertIsType('{}', dict)
        self.assertIsType('{x:y}', pentyping.Dict[int, str], env={'x': int,
                                                                 'y': str})
        self.assertIsType('{x:y, y:x}',
                          {pentyping.Dict[int, str], pentyping.Dict[str, int]},
                          env={'x': int, 'y': str})
        self.assertIsType('{x:y, y:x}',
                          pentyping.Dict[int, int],
                          env={'x': int, 'y': int})

    def test_set_ty(self):
        self.assertIsType('{x}', pentyping.Set[int], env={'x': int})
        self.assertIsType('{x, x}', pentyping.Set[int], env={'x': int})
        self.assertIsType('{x, y}',
                          {pentyping.Set[float], pentyping.Set[int]},
                          env={'x': int, 'y': float})

    def test_list_comp_ty(self):
        self.assertIsType('[x for x in y]', pentyping.List[str], env={'y': str})
        self.assertIsType('[x for x in y if 0]', list, env={'y': str})
        self.assertIsType('[1 for x in y if y]', pentyping.List[int], env={'y': str})
        self.assertIsType('[1 for x in y for z in y]', pentyping.List[int], env={'y': str})

    def test_set_comp_ty(self):
        self.assertIsType('{x for x in y}', pentyping.Set[str], env={'y': str})
        self.assertIsType('{x for x in y if 0}', set, env={'y': str})
        self.assertIsType('{1 for x in y if y}', pentyping.Set[int], env={'y': str})
        self.assertIsType('{1 for x in y for z in y}', pentyping.Set[int], env={'y': str})

    def test_dict_comp_ty(self):
        self.assertIsType('{x:1 for x in y}', pentyping.Dict[str, int], env={'y': str})
        self.assertIsType('{1:x for x in y if 0}', dict, env={'y': str})
        self.assertIsType('{1:1 for x in y if y}', pentyping.Dict[int, int], env={'y': str})
        self.assertIsType('{1:"" for x in y for z in y}', pentyping.Dict[int,str], env={'y': str})

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
        self.assertIsType('0 == 1 == x', pentyping.Cst[False], env={'x': int, 'y': int})
        self.assertIsType('0 == 0 == x', bool, env={'x': int, 'y': int})
        self.assertIsType('0 == 0 == 0', pentyping.Cst[True], env={'x': int, 'y': int})

        self.assertIsType('x < y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 < y', bool, env={'x': int, 'y': int})
        self.assertIsType('x < 1', bool, env={'x': int, 'y': int})
        self.assertIsType('0 < 1', pentyping.Cst[True], env={'x': int, 'y': int})
        self.assertIsType('2 < 1', pentyping.Cst[False], env={'x': int, 'y': int})

        self.assertIsType('x <= y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 <= y', bool, env={'x': int, 'y': int})
        self.assertIsType('x <= 1', bool, env={'x': int, 'y': int})
        self.assertIsType('1 <= 1', pentyping.Cst[True], env={'x': int, 'y': int})
        self.assertIsType('2 <= 1', pentyping.Cst[False], env={'x': int, 'y': int})

        self.assertIsType('x == y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 == y', bool, env={'x': int, 'y': int})
        self.assertIsType('x == 1', bool, env={'x': int, 'y': int})
        self.assertIsType('1 == 1', pentyping.Cst[True], env={'x': int, 'y': int})
        self.assertIsType('1 == 2', pentyping.Cst[False], env={'x': int, 'y': int})

        self.assertIsType('x != y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 != y', bool, env={'x': int, 'y': int})
        self.assertIsType('x != 1', bool, env={'x': int, 'y': int})
        self.assertIsType('1 != 1', pentyping.Cst[False], env={'x': int, 'y': int})
        self.assertIsType('1 != 2', pentyping.Cst[True], env={'x': int, 'y': int})

        self.assertIsType('x > y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 > y', bool, env={'x': int, 'y': int})
        self.assertIsType('x > 1', bool, env={'x': int, 'y': int})
        self.assertIsType('1 > 1', pentyping.Cst[False], env={'x': int, 'y': int})
        self.assertIsType('1 > 0', pentyping.Cst[True], env={'x': int, 'y': int})

        self.assertIsType('x >= y', bool, env={'x': int, 'y': int})
        self.assertIsType('1 >= y', bool, env={'x': int, 'y': int})
        self.assertIsType('x >= 1', bool, env={'x': int, 'y': int})
        self.assertIsType('0 >= 1', pentyping.Cst[False], env={'x': int, 'y': int})
        self.assertIsType('1 >= 0', pentyping.Cst[True], env={'x': int, 'y': int})

        self.assertIsType('x is y', bool, env={'x': int, 'y': int})
        self.assertIsType('x is y', pentyping.Cst[False], env={'x': int, 'y': float})
        self.assertIsType('1. is y', bool, env={'x': int, 'y': float})
        self.assertIsType('y is 1.', bool, env={'x': int, 'y': float})
        self.assertIsType('1 is y', pentyping.Cst[False], env={'x': int, 'y': float})
        self.assertIsType('y is 1', pentyping.Cst[False], env={'x': int, 'y': float})
        self.assertIsType('3 is 1', pentyping.Cst[False])
        self.assertIsType('3. is 1', pentyping.Cst[False])

        self.assertIsType('x is not y', bool, env={'x': int, 'y': int})
        self.assertIsType('x is not y', pentyping.Cst[True], env={'x': int, 'y': float})

        #self.assertIsType('x in [y]', bool, env={'x': int, 'y': int})
        #self.assertIsType('x not in [y]', bool, env={'x': int, 'y': int})

    def test_repr(self):
        self.assertIsType('repr(x)', str, env={'x': int})
        self.assertIsType('repr(1)', pentyping.Cst['1'], env={'x': int})

    def test_constant_ty(self):
        self.assertIsType('True', pentyping.Cst[True])
        self.assertIsType('False', pentyping.Cst[False])
        self.assertIsType('None', pentyping.Cst[None])
        self.assertIsType('1', pentyping.Cst[1])
        self.assertIsType('1.', pentyping.Cst[1.])
        self.assertIsType('1.j', pentyping.Cst[1.j])
        self.assertIsType('""', pentyping.Cst[""])

    def test_contant_op_ty(self):
        self.assertIsType('+(1)', pentyping.Cst[+1])
        self.assertIsType('-(1)', pentyping.Cst[-1])
        self.assertIsType('~1', pentyping.Cst[~1])
        self.assertIsType('not 1', pentyping.Cst[not 1])
        self.assertIsType('1 + 1', pentyping.Cst[1+1])
        self.assertIsType('1 - 1', pentyping.Cst[1 - 1])
        self.assertIsType('1 * 1', pentyping.Cst[1 * 1])
        self.assertIsType('1 << 1', pentyping.Cst[1 << 1])
        self.assertIsType('1 >> 1', pentyping.Cst[1 >> 1])
        #self.assertIsType('x @ x', int, env={"x": int})
        self.assertIsType('1 / 1', pentyping.Cst[1 / 1])
        self.assertIsType('1 % 1', pentyping.Cst[1 % 1])
        self.assertIsType('1 // 2', pentyping.Cst[1 // 2])
        self.assertIsType('1 | 2', pentyping.Cst[1 | 2])
        self.assertIsType('1 & 3', pentyping.Cst[1 & 3])
        self.assertIsType('2 ^ 3', pentyping.Cst[2 ^3])
        self.assertIsType('2 ** 3', pentyping.Cst[2 ** 3])

    def test_attr_ty(self):
        self.assertIsType('x.count("1")', int,
                          env={'x': pentyping.List[str]})
        self.assertIsType('dict.clear(x)', pentyping.Cst[None],
                          env={'x': pentyping.Dict[str, int]})

    def test_subscript_ty(self):
        self.assertIsType('x[y]', int, env={'x': pentyping.List[int], 'y': int})
        self.assertIsType('x[0]', int, env={'x': pentyping.List[int]})
        self.assertIsType('x[y]', pentyping.List[int],
                          env={'x': pentyping.List[int], 'y': slice})
        self.assertIsType('x[0:y]', pentyping.List[int],
                          env={'x': pentyping.List[int], 'y': int})
        self.assertIsType('x[0:1]', pentyping.List[int],
                          env={'x': pentyping.List[int]})
        self.assertIsType('x[::]', pentyping.List[int],
                          env={'x': pentyping.List[int]})

        self.assertIsType('x[y]', {int, float},
                          env={'x': pentyping.Tuple[int, float], 'y': int})
        self.assertIsType('x[0]', int, env={'x': pentyping.Tuple[int, float]})
        self.assertIsType('x[y]', tuple,
                          env={'x': pentyping.Tuple[int, float], 'y': slice})
        self.assertIsType('x[0:y]', tuple,
                          env={'x': pentyping.Tuple[int, float], 'y': int})
        self.assertIsType('x[0:1]', pentyping.Tuple[int],
                          env={'x': pentyping.Tuple[int, float]})
        self.assertIsType('x[::]', pentyping.Tuple[int, float],
                          env={'x': pentyping.Tuple[int, float]})

    def test_list_ty(self):
        self.assertIsType('[]', list)
        self.assertIsType('[1]', pentyping.List[int])
        self.assertIsType('[x]', pentyping.List[int], env={'x': int})
        self.assertIsType('[1, 1.]', {pentyping.List[int], pentyping.List[float]})

    def test_tuple_ty(self):
        self.assertIsType('()', tuple)
        self.assertIsType('(1,)', pentyping.Tuple[pentyping.Cst[1]])
        self.assertIsType('(1, x)', pentyping.Tuple[pentyping.Cst[1], int],
                          env={'x': int})
        self.assertIsType('(x, y)', pentyping.Tuple[int, float],
                          env={'x': int, 'y': float})
        self.assertIsType('(1, x or y)',
                          {pentyping.Tuple[pentyping.Cst[1], float],
                           pentyping.Tuple[pentyping.Cst[1], int]},
                          env={'x': int, 'y': float})

    def test_static_expression_ty(self):
        self.assertIsType('x is None',
                          pentyping.Cst[False],
                          env={'x': int})
        self.assertIsType('None is x',
                          pentyping.Cst[True],
                          env={'x': pentyping.Cst[None]})
        self.assertIsType('x is None or abs(x)',
                          {pentyping.FilteringBool[True, 'x', (pentyping.Cst[None],)],
                           int},
                          env={'x': {int, pentyping.Cst[None]}})
        self.assertIsType('not (x is None) and abs(x)',
                          {pentyping.FilteringBool[False, 'x', (pentyping.Cst[None],)],
                           int},
                          env={'x': {int, pentyping.Cst[None]}})
        self.assertIsType('x is not None',
                          pentyping.Cst[True],
                          env={'x': int})
        self.assertIsType('None is not x',
                          pentyping.Cst[False],
                          env={'x': pentyping.Cst[None]})
        self.assertIsType('x is not None and abs(x)',
                          {pentyping.FilteringBool[False, 'x', (pentyping.Cst[None],)],
                           int},
                          env={'x': {int, pentyping.Cst[None]}})
        self.assertIsType('not (type(x) is int) or 1',
                          {pentyping.FilteringBool[True, 'x', (pentyping.Cst[None],)],
                           pentyping.Cst[1]},
                          env={'x': {int, pentyping.Cst[None]}})
        self.assertIsType('type(x) is int and abs(x)',
                          {pentyping.FilteringBool[False, 'x', (pentyping.Cst[None],)],
                           int},
                          env={'x': {int, pentyping.Cst[None]}})
        self.assertIsType('x is None or None is y or (x + y)',
                          {pentyping.FilteringBool[True, 'x', (pentyping.Cst[None],)],
                           pentyping.FilteringBool[True, 'y', (pentyping.Cst[None],)],
                           int},
                          env={'x': {int, pentyping.Cst[None]},
                               'y': {int, pentyping.Cst[None]},
                              })
        self.assertIsType('x is y is None',
                          {pentyping.FilteringBool[False, 'x', (int,)],
                           pentyping.FilteringBool[True, 'x', (pentyping.Cst[None],)],
                           pentyping.Cst[True]},
                          env={'x': {int, pentyping.Cst[None]},
                               'y': {pentyping.Cst[None]},
                              })
        self.assertIsType('x is not None', pentyping.Cst[True], env={'x': int})
        self.assertIsType('x is None or x == 1', bool, env={'x': int})
        self.assertIsType('x is not None and x is not 2.', pentyping.Cst[True], env={'x': int})
        self.assertIsType('1 or x == 1', pentyping.Cst[1], env={'x': int})
        self.assertIsType('1 and x is not 2', bool, env={'x': int})
        self.assertIsType('1 and x is not 2.', pentyping.Cst[True], env={'x': int})
