from penty.types import Cst, Module, astype
from penty.types import Type, TypeOf, FilteringBool
from penty.types import ConstFunctionType, ConstFunctionTypeMeta
import operator
import re


class UnaryOperatorMeta(ConstFunctionTypeMeta):
    cache = {}

    def __getitem__(self, args):
        if args not in UnaryOperatorMeta.cache:
            class LocalUnaryOperator(UnaryOperator):
                __args__ = args,

            UnaryOperatorMeta.cache[args] = ConstFunctionType[LocalUnaryOperator,
                                                             getattr(operator, args)]
        return UnaryOperatorMeta.cache[args]

    def __repr__(self):
        return 'UnaryOperator[{}]'.format(self.__args__[0])

    def __call__(self, operand_ty):
        from penty.penty import Types
        return Types[operand_ty][self.__args__[0]](operand_ty)


class UnaryOperator(ConstFunctionType, metaclass=UnaryOperatorMeta):
    pass


class BinaryOperatorMeta(ConstFunctionTypeMeta):
    cache = {}

    def __getitem__(self, args):
        if args not in BinaryOperatorMeta.cache:
            assert re.match('^__[a-z]+__$', args)
            class LocalBinaryOperator(BinaryOperator):
                __args__ = args,

            BinaryOperatorMeta.cache[args] = ConstFunctionType[LocalBinaryOperator,
                                                              getattr(operator, args)]
        return BinaryOperatorMeta.cache[args]

    def __repr__(self):
        return 'BinaryOperator[{}]'.format(self.__args__[0])

    def __call__(self, left_ty, right_ty):
        from penty.penty import Types
        try:
            return Types[left_ty][self.__args__[0]](left_ty, right_ty)
        except TypeError:
            rop = '__r{}__'.format(self.__args__[0][2:-2])
            if rop not in Types[right_ty]:
                raise
            return Types[right_ty][rop](right_ty, left_ty)


class BinaryOperator(ConstFunctionType, metaclass=BinaryOperatorMeta):
    pass


class IsOperatorMeta(BinaryOperatorMeta):

    def __call__(self, left_ty, right_ty):
        if issubclass(left_ty, TypeOf):
            if issubclass(right_ty, (Cst, Type)):
                return FilteringBool[(left_ty.__args__[0] is right_ty.__args__[0]),
                                     left_ty.__args__[1], (left_ty.__args__[0],)]
            elif astype(left_ty) == right_ty:
                return bool
            else:
                return Cst[False]

        if issubclass(left_ty, (Cst, Type)):
            if issubclass(right_ty, (Cst, Type)):
                return Cst[left_ty.__args__[0] is right_ty.__args__[0]]
            elif astype(left_ty) == right_ty:
                return bool
            else:
                return Cst[False]
        else:
            left_ty = astype(left_ty)
            right_ty = astype(right_ty)
            if left_ty == right_ty:
                return bool
            else:
                return Cst[False]


class IsOperator(BinaryOperator, metaclass=IsOperatorMeta):
    __args__ = "is",


IsOperator = ConstFunctionType[IsOperator, operator.is_]


class IsNotOperatorMeta(BinaryOperatorMeta):

    def __call__(self, left_ty, right_ty):
        if issubclass(left_ty, Cst):
            if issubclass(right_ty, Cst):
                return Cst[left_ty.__args__[0] is not right_ty.__args__[0]]
            elif astype(left_ty) == right_ty:
                return bool
            else:
                return Cst[True]
        else:
            left_ty = astype(left_ty)
            right_ty = astype(right_ty)
            if left_ty == right_ty:
                return bool
            else:
                return Cst[True]


class IsNotOperator(BinaryOperator, metaclass=IsNotOperatorMeta):
    __args__ = "is not",


IsNotOperator = ConstFunctionType[IsNotOperator, operator.is_]


class NotOperatorMeta(UnaryOperatorMeta):

    def __call__(self, operand_ty):
        from penty.penty import Types
        if '__not__' in Types[operand_ty]:
            func = Types[operand_ty]['__not__']
        else:
            def func(argument_ty):
                bool_ty = Types[argument_ty]['__bool__'](argument_ty)
                return Types[bool_ty]['__not__'](bool_ty)

        return func(operand_ty)


class NotOperator(UnaryOperator, metaclass=NotOperatorMeta):
    __args__ = "not",


NotOperator = ConstFunctionType[NotOperator, operator.not_]


def register(registry):
    if Module['operator'] not in registry:
        registry[Module['operator']] = {
            '__add__': BinaryOperator['__add__'],
            '__and__': BinaryOperator['__and__'],
            '__eq__': BinaryOperator['__eq__'],
            '__floordiv__': BinaryOperator['__floordiv__'],
            '__ge__': BinaryOperator['__ge__'],
            '__getitem__': BinaryOperator['__getitem__'],
            '__gt__': BinaryOperator['__gt__'],
            '__iadd__': BinaryOperator['__iadd__'],
            '__iand__': BinaryOperator['__iand__'],
            '__ior__': BinaryOperator['__ior__'],
            '__ixor__': BinaryOperator['__ixor__'],
            '__itruediv__': BinaryOperator['__itruediv__'],
            '__ifloordiv__': BinaryOperator['__ifloordiv__'],
            '__imatmul__': BinaryOperator['__imatmul__'],
            '__imod__': BinaryOperator['__imod__'],
            '__imul__': BinaryOperator['__imul__'],
            '__ipow__': BinaryOperator['__ipow__'],
            '__isub__': BinaryOperator['__isub__'],
            '__invert__': UnaryOperator['__invert__'],
            '__le__': BinaryOperator['__le__'],
            '__lshift__': BinaryOperator['__lshift__'],
            '__lt__': BinaryOperator['__lt__'],
            '__matmul__': BinaryOperator['__matmul__'],
            '__mod__': BinaryOperator['__mod__'],
            '__mul__': BinaryOperator['__mul__'],
            '__ne__': BinaryOperator['__ne__'],
            '__neg__': UnaryOperator['__neg__'],
            '__not__': NotOperator,
            '__or__': BinaryOperator['__or__'],
            '__pos__': UnaryOperator['__pos__'],
            '__pow__': BinaryOperator['__pow__'],
            '__rshift__': BinaryOperator['__rshift__'],
            '__sub__': BinaryOperator['__sub__'],
            '__truediv__': BinaryOperator['__truediv__'],
            '__xor__': BinaryOperator['__xor__'],
            'add': BinaryOperator['__add__'],
            'and_': BinaryOperator['__and__'],
            'eq': BinaryOperator['__eq__'],
            'floordiv': BinaryOperator['__floordiv__'],
            'ge': BinaryOperator['__ge__'],
            'getitem': BinaryOperator['__getitem__'],
            'gt': BinaryOperator['__gt__'],
            'iadd': BinaryOperator['__iadd__'],
            'iand': BinaryOperator['__iand__'],
            'ior': BinaryOperator['__ior__'],
            'ixor': BinaryOperator['__ixor__'],
            'itruediv': BinaryOperator['__itruediv__'],
            'ifloordiv': BinaryOperator['__ifloordiv__'],
            'imatmul': BinaryOperator['__imatmul__'],
            'imod': BinaryOperator['__imod__'],
            'imul': BinaryOperator['__imul__'],
            'ipow': BinaryOperator['__ipow__'],
        'isub': BinaryOperator['__isub__'],
        'invert': UnaryOperator['__invert__'],
        'le': BinaryOperator['__le__'],
        'lshift': BinaryOperator['__lshift__'],
        'lt': BinaryOperator['__lt__'],
        'matmul': BinaryOperator['__matmul__'],
        'mod': BinaryOperator['__mod__'],
        'mul': BinaryOperator['__mul__'],
        'ne': BinaryOperator['__ne__'],
        'neg': UnaryOperator['__neg__'],
        'not_': NotOperator,
        'or_': BinaryOperator['__or__'],
        'pos': UnaryOperator['__pos__'],
        'pow': BinaryOperator['__pow__'],
        'rshift': BinaryOperator['__rshift__'],
        'sub': BinaryOperator['__sub__'],
        'truediv': BinaryOperator['__truediv__'],
        'xor': BinaryOperator['__xor__'],
        }
