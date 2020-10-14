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
