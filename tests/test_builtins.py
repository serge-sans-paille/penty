from unittest import TestCase
import gast as ast
import penty
import penty.types as pentyping
import typing

class TestBuiltins(TestCase):

    def assertIsType(self, expr, ty, env={}):
        if not isinstance(ty, set):
            ty = {ty}
        env = {k: v if isinstance(v, set) else {v}
               for k, v in env.items()}
        self.assertEqual(penty.type_eval(expr, env), ty)

    def test_type(self):
        self.assertIsType('type(x) is int', pentyping.Cst[True], env={'x':int})

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


class TestDict(TestCase):

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

class TestList(TestCase):

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
