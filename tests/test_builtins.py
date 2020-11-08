from unittest import TestCase
import gast as ast
import penty
import penty.types as pentyping
import typing

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
        self.assertTypesEqual(penty.type_eval(expr, env), ty)


class TestBuiltins(TestPenty):

    def test_type(self):
        self.assertIsType('type(x) is int',
                          pentyping.FilteringBool[True, 'x', (int,)],
                          env={'x':int})
        self.assertIsType('abs(x) if type(x) is int else x',
                          {int, pentyping.Cst[None]},
                          env={'x': {int, pentyping.Cst[None]}})
        self.assertIsType('type(x)(x)', int, env={'x': int})

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


class TestDict(TestPenty):

    def test_clear(self):
        self.assertIsType('x.clear()', pentyping.Cst[None],
                          env={'x': pentyping.Dict[int, int]})

    def test_from_keys(self):
        self.assertIsType('dict.from_keys(x)',
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


class TestList(TestPenty):

    def test_append(self):
        self.assertIsType('x.append(y), x',
                          pentyping.Tuple[pentyping.Cst[None], pentyping.List[int]],
                          env={'x': pentyping.List[set()], 'y': int})
        self.assertIsType('(x.append(y), x)[1]',
                          {pentyping.List[{int, float}]},
                          env={'x': pentyping.List[float], 'y': int})


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

class TestStr(TestPenty):

    def test_iter(self):
        self.assertIsType('x.__iter__().__next__()', str, env={'x': str})
