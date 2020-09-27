from unittest import TestCase
from textwrap import dedent
import gast as ast
import penty
import penty.types as pentyping
import typing

class TestStmt(TestCase):

    def assertIsType(self, stmt, expr, ty, env={}):
        if not isinstance(ty, set):
            ty = {ty}
        type_env = penty.type_exec(dedent(stmt), {})
        type_env.update({k: v if isinstance(v, set) else {v}
                         for k, v in env.items()})
        self.assertEqual(penty.type_eval(expr, type_env), ty)

    def test_fdef_noarg_ty(self):
        self.assertIsType('def func(): pass',
                          'func()', pentyping.Cst[None])

    def test_fdef_forward_ty(self):
        self.assertIsType('def func(x): return x',
                          'func(x)', int,
                          env={'x':int})

    def test_fdef_forward_multiarg_ty(self):
        self.assertIsType('def func(x, y): return x, y',
                          'func(x, x)', typing.Tuple[int, int],
                          env={'x':int})
