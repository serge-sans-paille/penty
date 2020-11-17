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

    def test_fdef_kwarg_ty(self):
        self.assertIsType('def func(y): return y',
                          'func(y=x)', int,
                          env={'x':int})

    def test_fdef_forward_multiarg_ty(self):
        self.assertIsType('def func(x, y): return x, y',
                          'func(x, x)', pentyping.Tuple[int, int],
                          env={'x':int})

    def test_fdef_kw_multiarg_ty(self):
        self.assertIsType('def func(x, y): return x, y',
                          'func(x, y=1.)',
                          pentyping.Tuple[int, pentyping.Cst[1.]],
                          env={'x':int})

    def test_fdef_kwonly_multiarg_ty(self):
        self.assertIsType('def func(*, x, y): return x, y',
                          'func(y=1., x=x)',
                          pentyping.Tuple[int, pentyping.Cst[1.]],
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
                          'x, y', pentyping.Tuple[pentyping.Cst[1],
                                               pentyping.Cst[2]])

    def test_type_destructuring_assign_ex(self):
        self.assertIsType('x, (y, z) = 1, (2, "3")',
                          'x, y, z', pentyping.Tuple[pentyping.Cst[1],
                                                  pentyping.Cst[2],
                                                  pentyping.Cst["3"]])

    def test_multi_assign(self):
        self.assertIsType('x = y = 1',
                          'x, y', pentyping.Tuple[pentyping.Cst[1],
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

    def test_for_loop_simple_incr(self):
        self.assertIsType('j = 0\nfor i in "hello": j += 1',
                          'j', int)

    def test_for_early_return(self):
        self.assertIsType('j = 0\nfor i in "hello":\n j += 1\n return',
                          'j', {pentyping.Cst[0], pentyping.Cst[1]})

    def test_for_early_break(self):
        self.assertIsType('j = 0\nfor i in "hello":\n j += 1\n break',
                          'j', {pentyping.Cst[0], pentyping.Cst[1]})

    def test_for_early_continue(self):
        self.assertIsType('j = 0\nfor i in "hello":\n j += 1\n continue',
                          'j', int)

    def test_for_early_return_else(self):
        self.assertIsType('j = 0\nfor i in "hello": return\nelse: j = 1',
                          'j', pentyping.Cst[0])

    def test_for_early_break_else(self):
        self.assertIsType('j = 0\nfor i in "hello":\n break\nelse: j = 1',
                          'j', pentyping.Cst[0])

    def test_for_early_continue_else(self):
        self.assertIsType('j = 0\nfor i in "hello":\n continue\nelse: j = 1',
                          'j', pentyping.Cst[1])

    def test_while_loop_trivial_false(self):
        self.assertIsType('i = 0\nwhile 0: i = 1',
                          'i', pentyping.Cst[0])

    def test_while_loop_trivial_true(self):
        self.assertIsType('i = 0\nwhile 1: i = 1',
                          'i', pentyping.Cst[1])

    def test_while_loop_trivial_inifinte(self):
        self.assertIsType('i = 0\nwhile 1: i = 1\ni = 2',
                          'i', pentyping.Cst[1])

    def test_while_loop_simple(self):
        self.assertIsType('i = 0\nwhile id(i): j = i',
                          'j', pentyping.Cst[0])

    def test_while_loop_simple_incr(self):
        self.assertIsType('j = 0\nwhile id(j): j += 1',
                          'j', {pentyping.Cst[0], int})

    def test_while_early_return(self):
        self.assertIsType('j = 0\nwhile id(j):\n j += 1\n return',
                          'j', {pentyping.Cst[0], pentyping.Cst[1]})

    def test_while_early_break(self):
        self.assertIsType('j = 0\nwhile id(j):\n j += 1\n break',
                          'j', {pentyping.Cst[0], pentyping.Cst[1]})

    def test_while_early_continue(self):
        self.assertIsType('j = 0\nwhile id(j):\n j += 1\n continue',
                          'j', {pentyping.Cst[0], int})

    def test_while_early_return_else(self):
        self.assertIsType('j = 0\nwhile id(j): return\nelse: j = 1',
                          'j', pentyping.Cst[0])

    def test_while_early_break_else(self):
        self.assertIsType('j = 0\nwhile id(j):\n break\nelse: j = 1',
                          'j', pentyping.Cst[0])

    def test_while_early_continue_else(self):
        self.assertIsType('j = 0\nwhile id(j):\n continue\nelse: j = 1',
                          'j', {pentyping.Cst[0], pentyping.Cst[1]})

    def test_while_isnone(self):
        self.assertIsType('i = id(None) or None\nj=1.\nwhile i is None: j = i',
                          'j', {pentyping.Cst[1.], pentyping.Cst[None]})


    def test_if_true_branch(self):
        self.assertIsType('j = 0\nif j != 1: j = 1',
                          'j', pentyping.Cst[1])

    def test_if_false_branch(self):
        self.assertIsType('j = 0\nif j != 0: j = 1\nelse: j = 2',
                          'j', pentyping.Cst[2])

    def test_if_false_empty_branch(self):
        self.assertIsType('j = 0\nif j != 0: j = 1\nj = 3',
                          'j', pentyping.Cst[3])

    def test_if_both(self):
        self.assertIsType('j = 0\nif id(j): j = 1\nelse: j = 3',
                          'j', {pentyping.Cst[1], pentyping.Cst[3]})

    def test_if_body(self):
        self.assertIsType('j = 0\nif id(j): j = 1',
                          'j', {pentyping.Cst[0], pentyping.Cst[1]})

    def test_if_else(self):
        self.assertIsType('j = 0\nif id(j): pass\nelse: j = 1',
                          'j', {pentyping.Cst[0], pentyping.Cst[1]})

    def test_if_else_in_loop_assign(self):
        code = '''
        for j in "erty":
            if j: v = 1
            else: s = v'''
        self.assertIsType(code,
                          'v, s',
                          pentyping.Tuple[pentyping.Cst[1], pentyping.Cst[1]])

    def test_if_else_in_loop_break(self):
        code = '''
        for j in "erty":
            if j:
                s = j
                break
        else:
            s = 1'''
        self.assertIsType(code,
                          's',
                          {pentyping.Cst[1], str})

    def test_if_return_body_default(self):
        code = '''
        def f(x):
            if x:
                return x'''
        self.assertIsType(code,
                          'f(y)', {int, pentyping.Cst[None]},
                          env = {'y': int})

    def test_if_return_body(self):
        code = '''
        def f(x):
            if x:
                return x
            return "e"'''
        self.assertIsType(code,
                          'f(y)', {int, pentyping.Cst["e"]},
                          env = {'y': int})

    def test_if_return_orelse_default(self):
        code = '''
        def f(x):
            if x:
                pass
            else:
                return x'''
        self.assertIsType(code,
                          'f(y)', {int, pentyping.Cst[None]},
                          env = {'y': int})

    def test_if_return_orelse(self):
        code = '''
        def f(x):
            if x:
                pass
            else:
                return x
            return "e"'''
        self.assertIsType(code,
                          'f(y)', {int, pentyping.Cst["e"]},
                          env = {'y': int})

    def test_if_return(self):
        code = '''
        def f(x):
            if x:
                return "x"
            else:
                return x'''
        self.assertIsType(code,
                          'f(y)', {int, pentyping.Cst["x"]},
                          env = {'y': int})

    def test_nested_if_return(self):
        code = '''
        def f(x):
            if x:
                if id(x):
                    return "x"
            else:
                return x'''
        self.assertIsType(code,
                          'f(y)', {int, pentyping.Cst["x"], pentyping.Cst[None]},
                          env = {'y': int})

    def test_if_isnone(self):
        code = '''
        def f(x, y):
            if x is None:
                return y
            else:
                return abs(x)'''
        self.assertIsType(code,
                          'f(x, y)', float,
                          env = {'x': {pentyping.Cst[None], float},
                                 'y': float})

    def test_import(self):
        self.assertIsType('import operator; x = operator.add(1, 2)',
                          'x', pentyping.Cst[3])

    def test_import_as(self):
        self.assertIsType('import operator as op; x = op.add(1, 2)',
                          'x', pentyping.Cst[3])

    def test_import_package(self):
        self.assertIsType('import numpy.random; x = numpy.random.bytes(2)',
                          'x', str)

    def test_import_package_as(self):
        self.assertIsType('import numpy.random as random; x = random.bytes(2)',
                          'x', str)

    def test_import_from(self):
        self.assertIsType('from operator import add; x = add(1, 2)',
                          'x', pentyping.Cst[3])

    def test_import_from_as(self):
        self.assertIsType('from operator import add as op; x = op(1, 2)',
                          'x', pentyping.Cst[3])

    def test_import_from_package(self):
        self.assertIsType('from numpy.random import bytes; x = bytes(1)',
                          'x', str)

