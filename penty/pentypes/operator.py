from penty.types import FunctionType, FunctionTypeMeta, Cst, Module, astype

class UnaryOperatorMeta(FunctionTypeMeta):
    cache = {}

    def __getitem__(self, args):
        if args not in UnaryOperatorMeta.cache:
            class LocalUnaryOperator(UnaryOperator):
                __args__ = args,

            UnaryOperatorMeta.cache[args] = Cst[LocalUnaryOperator]
        return UnaryOperatorMeta.cache[args]

    def __repr__(self):
        return 'UnaryOperator[{}]'.format(self.__args__[0])

    def __call__(self, operand_types):
        from penty.penty import Types
        result_type = set()
        for operand_type in operand_types:
            result_type.update(Types[operand_type][self.__args__[0]](operand_types))
        return result_type


class UnaryOperator(FunctionType, metaclass=UnaryOperatorMeta):
    pass

class BinaryOperatorMeta(FunctionTypeMeta):
    cache = {}

    def __getitem__(self, args):
        if args not in BinaryOperatorMeta.cache:
            class LocalBinaryOperator(BinaryOperator):
                __args__ = args,

            BinaryOperatorMeta.cache[args] = Cst[LocalBinaryOperator]
        return BinaryOperatorMeta.cache[args]

    def __repr__(self):
        return 'BinaryOperator[{}]'.format(self.__args__[0])

    def __call__(self, left_types, right_types):
        from penty.penty import Types
        result_type = set()
        for left_ty in list(left_types):
            result_type.update(Types[left_ty][self.__args__[0]]
                               (left_types, right_types))
        return result_type


class BinaryOperator(FunctionType, metaclass=BinaryOperatorMeta):
    pass

class IsOperatorMeta(BinaryOperatorMeta):

    def __call__(self, left_types, right_types):
        result_types = set()
        for left_ty in left_types:
            if issubclass(left_ty, Cst):
                for right_ty in right_types:
                    if issubclass(right_ty, Cst):
                        result_types.add(Cst[left_ty.__args__[0] is
                                              right_ty.__args__[0]])
                    elif astype(left_ty) == right_ty:
                        result_types.add(bool)
                    else:
                        result_types.add(Cst[False])
            else:
                left_ty = astype(left_ty)
                for right_ty in right_types:
                    right_ty = astype(right_ty)
                    if left_ty == right_ty:
                        result_types.add(bool)
                    else:
                        result_types.add(Cst[False])
        return result_types

class IsOperator(BinaryOperator, metaclass=IsOperatorMeta):
    __args__ = "is",

IsOperator = Cst[IsOperator]

class IsNotOperatorMeta(BinaryOperatorMeta):

    def __call__(self, left_types, right_types):
        result_types = set()
        for left_ty in left_types:
            if issubclass(left_ty, Cst):
                for right_ty in right_types:
                    if issubclass(right_ty, Cst):
                        result_types.add(Cst[left_ty.__args__[0] is not
                                              right_ty.__args__[0]])
                    elif astype(left_ty) == right_ty:
                        result_types.add(bool)
                    else:
                        result_types.add(Cst[True])
            else:
                left_ty = astype(left_ty)
                for right_ty in right_types:
                    right_ty = astype(right_ty)
                    if left_ty == right_ty:
                        result_types.add(bool)
                    else:
                        result_types.add(Cst[True])
        return result_types

class IsNotOperator(BinaryOperator, metaclass=IsNotOperatorMeta):
    __args__ = "is not",

IsNotOperator = Cst[IsNotOperator]

class NotOperatorMeta(UnaryOperatorMeta):

    def __call__(self, operand_types):
        from penty.penty import Types
        result_type = set()
        for operand_type in operand_types:
            if '__not__' in Types[operand_type]:
                func = Types[operand_type]['__not__']
            else:
                def func(argument_types):
                    result_types = set()
                    for z in argument_types:
                        z_bool_ty = Types[z]['__bool__'](argument_types)
                        for y in z_bool_ty:
                            result_types.update(Types[y]['__not__'](z_bool_ty))
                    return result_types

            result_type.update(func(operand_types))
        return result_type

class NotOperator(UnaryOperator, metaclass=NotOperatorMeta):
    __args__ = "not",

NotOperator = Cst[NotOperator]

def register(registry):
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
    'sub': BinaryOperator['__sub__'],
    'truediv': BinaryOperator['__truediv__'],
    'xor': BinaryOperator['__xor__'],
    }
