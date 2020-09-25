from unittest import TestCase
import gast as ast
import penty

class TestExpr(TestCase):

    def assertIsType(self, expr, ty, env={}):
        if not isinstance(ty, set):
            ty = {ty}
        self.assertEqual(penty.type_eval(expr, env), ty)

    def test_constant_ty(self):
        self.assertIsType('True', bool)
        self.assertIsType('False', bool)
        self.assertIsType('None', type(None))
        self.assertIsType('1', int)
        self.assertIsType('1.', float)
        self.assertIsType('1.j', complex)
        self.assertIsType('""', str)

    def test_unaryop_ty(self):
        self.assertIsType('+(1)', int)
        self.assertIsType('-(1)', int)
        self.assertIsType('~(1)', int)

    def test_binaryop_ty(self):
        self.assertIsType('1 + 1', int)
        self.assertIsType('1 - 1', int)
        self.assertIsType('1 * 1', int)
        #self.assertIsType('1 @ 1', int)
        self.assertIsType('1 / 1', float)
        self.assertIsType('1 % 1', int)
        self.assertIsType('1 // 1', int)
        self.assertIsType('1 | 1', int)
        self.assertIsType('1 & 1', int)
        self.assertIsType('1 ^ 1', int)
        self.assertIsType('1 ** 1', int)
