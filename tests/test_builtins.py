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
