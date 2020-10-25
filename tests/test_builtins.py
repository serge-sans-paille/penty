from unittest import TestCase
import gast as ast
import penty
import penty.types as pentyping
import typing

class TestPenty(TestCase):

    def assertIsType(self, expr, ty, env={}):
        if not isinstance(ty, set):
            ty = {ty}
        env = {k: v if isinstance(v, set) else {v}
               for k, v in env.items()}
        self.assertEqual(penty.type_eval(expr, env), ty)


class TestBuiltins(TestPenty):

    def test_type(self):
        self.assertIsType('type(x) is int', pentyping.Cst[True], env={'x':int})

    def test_abs(self):
        self.assertIsType('abs(x)', int, env={'x': bool})
        self.assertIsType('abs(x)', int, env={'x': int})
        self.assertIsType('abs(x)', float, env={'x': float})

    def test_float(self):
        self.assertIsType('float(x)', float, env={'x': bool})
        self.assertIsType('float(x)', float, env={'x': int})
        self.assertIsType('float(x)', float, env={'x': float})
        self.assertIsType('float(x)', float, env={'x': str})

    def test_int(self):
        self.assertIsType('int(x)', int, env={'x': bool})
        self.assertIsType('int(x)', int, env={'x': int})
        self.assertIsType('int(x)', int, env={'x': float})
        self.assertIsType('int(x)', int, env={'x': str})

    def test_len(self):
        self.assertIsType('len(x)', pentyping.Cst[2],
                          env={'x': typing.Tuple[int, int]})
        self.assertIsType('len(x)', int,
                          env={'x': typing.List[int]})
        self.assertIsType('len(x)', int,
                          env={'x': typing.Set[int]})
        self.assertIsType('len(x)', int,
                          env={'x': typing.Dict[int, str]})
        self.assertIsType('len(x)', int,
                          env={'x': str})
        self.assertIsType('len("hello")', pentyping.Cst[5])

    def test_bool(self):
        self.assertIsType('bool(True)', pentyping.Cst[True])
        self.assertIsType('bool(False)', pentyping.Cst[False])
        self.assertIsType('bool(x)', bool, env={'x': bool})
        self.assertIsType('bool(x)', bool,
                          env={'x': int})
        self.assertIsType('bool(x)', bool,
                          env={'x': float})
        self.assertIsType('bool(x)', pentyping.Cst[True],
                          env={'x': typing.Tuple[int, int]})
        self.assertIsType('bool(2)', pentyping.Cst[True])
        self.assertIsType('bool(x)', bool,
                          env={'x': typing.List[int]})
        self.assertIsType('bool(x)', bool,
                          env={'x': typing.Set[int]})
        self.assertIsType('bool(x)', bool,
                          env={'x': typing.Dict[int, str]})
        self.assertIsType('bool(x)', bool,
                          env={'x': str})
        self.assertIsType('bool("")', pentyping.Cst[False])
        self.assertIsType('bool("hello")', pentyping.Cst[True])

class TestFloat(TestPenty):

    def test_abs(self):
        self.assertIsType('abs(-1.)', pentyping.Cst[abs(-1.)])
        self.assertIsType('abs(x)', float, env={'x': float})

    def test_add(self):
        self.assertIsType('1. + 3.4', pentyping.Cst[1 + 3.4])
        self.assertIsType('x + 3.4', float, env={'x': float})
        self.assertIsType('x + x', float, env={'x': float})

    def test_bool(self):
        self.assertIsType('bool(1.)', pentyping.Cst[bool(1.)])
        self.assertIsType('bool(x)', bool, env={'x': float})

    def test_eq(self):
        self.assertIsType('1. == 3.4', pentyping.Cst[1 == 3.4])
        self.assertIsType('x == 3.4', bool, env={'x': float})
        self.assertIsType('x == x', bool, env={'x': float})

    def test_floordiv(self):
        self.assertIsType('1. // 3.4', pentyping.Cst[1 // 3.4])
        self.assertIsType('x // 3.4', float, env={'x': float})
        self.assertIsType('x // x', float, env={'x': float})

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

    def test_mod(self):
        self.assertIsType('1. % 3.4', pentyping.Cst[1 % 3.4])
        self.assertIsType('x % 3.4', float, env={'x': float})
        self.assertIsType('x % x', float, env={'x': float})

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

    def test_sub(self):
        self.assertIsType('1. - 3.4', pentyping.Cst[1 - 3.4])
        self.assertIsType('x - 3.4', float, env={'x': float})
        self.assertIsType('x - x', float, env={'x': float})


class TestDict(TestPenty):

    def assertIsType(self, expr, ty, env={}):
        if not isinstance(ty, set):
            ty = {ty}
        env = {k: v if isinstance(v, set) else {v}
               for k, v in env.items()}
        self.assertEqual(penty.type_eval(expr, env), ty)

    def test_clear(self):
        self.assertIsType('x.clear()', pentyping.Cst[None],
                          env={'x':typing.Dict[int, int]})

    def test_from_keys(self):
        self.assertIsType('dict.from_keys(x)',
                          typing.Dict[str, pentyping.Cst[None]],
                          env={'x': str})

class TestList(TestPenty):

    def assertIsType(self, expr, ty, env={}):
        if not isinstance(ty, set):
            ty = {ty}
        env = {k: v if isinstance(v, set) else {v}
               for k, v in env.items()}
        self.assertEqual(penty.type_eval(expr, env), ty)

    def test_append(self):
        self.assertIsType('x.append(y), x',
                          typing.Tuple[pentyping.Cst[None], typing.List[int]],
                          env={'x': list, 'y': int})
        self.assertIsType('(x.append(y), x)[1]',
                          {typing.List[int], typing.List[float]},
                          env={'x': typing.List[float], 'y': int})
