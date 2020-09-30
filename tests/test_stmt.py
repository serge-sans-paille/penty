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

    def test_recursive_function_ty(self):
        code = '''
            def fibo(x):
                return x if x < 2 else fibo(x-1) + fibo(x-2)'''
        self.assertIsType(code,
                          'fibo(n)', int,
                          env={'n':int})

    def test_recursive_function_cst_ty(self):
        code = '''
            def fibo(x):
                return x if x < 2 else fibo(x-1) + fibo(x-2)'''
        self.assertIsType(code,
                          'fibo(n)', pentyping.Cst[8],
                          env={'n': pentyping.Cst[6]})

    def test_del_single(self):
        stmt = 'del a'
        env = {'a': int}
        type_env = penty.type_exec(dedent(stmt), env)
        self.assertNotIn('a', type_env)
        with self.assertRaises(penty.penty.UnboundIdentifier):
            stmt = 'del a; a'
            penty.type_exec(dedent(stmt), env)

    def test_del_multiple(self):
        stmt = 'del a, b'
        env = {'a': int, 'b': float}
        type_env = penty.type_exec(dedent(stmt), env)
        self.assertNotIn('a', type_env)
        self.assertNotIn('b', type_env)
        with self.assertRaises(penty.penty.UnboundIdentifier):
            stmt = 'del a, b; a, b'
            penty.type_exec(dedent(stmt), env)

    def test_assign(self):
        self.assertIsType('x = 1',
                          'x', pentyping.Cst[1])

    def test_type_destructuring_assign(self):
        self.assertIsType('x, y = 1, 2',
                          'x, y', typing.Tuple[pentyping.Cst[1],
                                               pentyping.Cst[2]])

    def test_type_destructuring_assign_ex(self):
        self.assertIsType('x, (y, z) = 1, (2, "3")',
                          'x, y, z', typing.Tuple[pentyping.Cst[1],
                                                  pentyping.Cst[2],
                                                  pentyping.Cst["3"]])

    def test_multi_assign(self):
        self.assertIsType('x = y = 1',
                          'x, y', typing.Tuple[pentyping.Cst[1],
                                               pentyping.Cst[1]])

    def test_reassign(self):
        self.assertIsType('x = 1; x = 1.',
                          'x', pentyping.Cst[1.])

    def test_update_operators(self):
        for op in ('+', '&', '|', '^', '/', '//', '%', '*', '**', '-'):
            self.assertIsType('x = 2; x {}= 3'.format(op),
                              'x',
                              pentyping.Cst[eval("2 {} 3".format(op))])

    def test_for_loop_simple(self):
        self.assertIsType('for i in "hello": pass',
                          'i', str)

    def test_for_loop_simple(self):
        self.assertIsType('j = 0\nfor i in "hello": j += 1',
                          'j', int)
