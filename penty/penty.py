import gast as ast
import typing
import itertools

import penty.pentypes as pentypes
from penty.types import Cst




class UnboundIdentifier(RuntimeError):
    pass

class OperatorModule(object):
    pass


class Lambda(object):

    def __init__(self, node, visitor):
        self.node = node
        self.visitor = visitor
        self.bindings = list(visitor.bindings)

    def __call__(self, *argument_types):
        old_bindings, self.visitor.bindings = self.visitor.bindings, self.bindings
        new_bindings = {arg.id: arg_ty for arg, arg_ty in
                        zip(self.node.args.args, argument_types)}
        self.visitor.bindings.append(new_bindings)
        result_types = self.visitor.visit(self.node.body)
        self.visitor.bindings.pop()
        self.visitor.bindings = old_bindings
        return result_types

class BinaryOperator(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, left_types, right_types):
        result_type = set()
        for left_ty in left_types:
            result_type.update(Types[left_ty][self.name]
                               (left_types, right_types))
        return result_type

def IsOperator(left_types, right_types):
    result_types = set()
    for left_ty in left_types:
        left_ty = astype(left_ty)
        for right_ty in right_types:
            right_ty = astype(right_ty)
            if left_ty == right_ty:
                result_types.add(bool)
            else:
                result_types.add(Cst[False])
    return result_types

def IsNotOperator(left_types, right_types):
    result_types = set()
    for left_ty in left_types:
        left_ty = astype(left_ty)
        for right_ty in right_types:
            right_ty = astype(right_ty)
            if left_ty == right_ty:
                result_types.add(bool)
            else:
                result_types.add(Cst[True])
    return result_types

class UnaryOperator(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, operand_types):
        result_type = set()
        for operand_type in operand_types:
            result_type.update(Types[operand_type][self.name](operand_types))
        return result_type

class NotOperator(object):
    def __call__(self, operand_types):
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

class TypeRegistry(object):

    def __init__(self, registry):
        self.registry = registry

    def __getattr__(self, attr):
        return self.registry.attr

    def __getitem__(self, key):
        if issubclass(key, Cst):
            return self.registry[type(key.__args__[0])]
        if issubclass(key, (typing.List, typing.Set, typing.Dict,
                            typing.Tuple)):
            self.registry[key] = self.instanciate(key)
        return self.registry[key]

    def instanciate(self, ty):
        if ty in self.registry:
            return self.registry[ty]
        base = ty.__bases__[0]
        return self.registry[base].instanciate(ty)



Types = TypeRegistry({
    bool : {
        '__not__': pentypes.bool.not_,
    },
    int: {
        '__add__': pentypes.int.add,
        '__and__': pentypes.int.bitand,
        '__bool__': pentypes.int.boolean,
        '__eq__': pentypes.int.eq,
        '__floordiv__': pentypes.int.floordiv,
        '__ge__': pentypes.int.ge,
        '__gt__': pentypes.int.gt,
        '__invert__': pentypes.int.invert,
        '__le__': pentypes.int.le,
        '__lt__': pentypes.int.lt,
        '__mul__': pentypes.int.mul,
        '__mod__': pentypes.int.mod,
        '__ne__': pentypes.int.ne,
        '__neg__': pentypes.int.neg,
        '__or__': pentypes.int.bitor,
        '__pos__': pentypes.int.pos,
        '__pow__': pentypes.int.power,
        '__sub__': pentypes.int.sub,
        '__truediv__': pentypes.int.truediv,
        '__xor__': pentypes.int.bitxor,
    },
    str: {
        '__iter__': pentypes.str.iterator,
    },
    pentypes.str.str_iterator: {
        '__next__': pentypes.str_iterator.next_element,
    },
    OperatorModule: {
        '__add__': BinaryOperator('__add__'),
        '__and__': BinaryOperator('__and__'),
        '__eq__': BinaryOperator('__eq__'),
        '__floordiv__': BinaryOperator('__floordiv__'),
        '__ge__': BinaryOperator('__ge__'),
        '__getitem__': BinaryOperator('__getitem__'),
        '__gt__': BinaryOperator('__gt__'),
        '__invert__': UnaryOperator('__invert__'),
        '__le__': BinaryOperator('__le__'),
        '__lt__': BinaryOperator('__lt__'),
        '__matmul__': BinaryOperator('__matmul__'),
        '__mod__': BinaryOperator('__mod__'),
        '__mul__': BinaryOperator('__mul__'),
        '__ne__': BinaryOperator('__ne__'),
        '__neg__': UnaryOperator('__neg__'),
        '__not__': NotOperator(),
        '__or__': BinaryOperator('__or__'),
        '__pos__': UnaryOperator('__pos__'),
        '__pow__': BinaryOperator('__pow__'),
        '__sub__': BinaryOperator('__sub__'),
        '__truediv__': BinaryOperator('__truediv__'),
        '__xor__': BinaryOperator('__xor__'),
    },
    typing.List: pentypes.list,
    typing.Tuple: pentypes.tuple,
    })

Ops = {
    ast.Add: Types[OperatorModule]['__add__'],
    ast.BitAnd: Types[OperatorModule]['__and__'],
    ast.BitOr: Types[OperatorModule]['__or__'],
    ast.BitXor: Types[OperatorModule]['__xor__'],
    ast.Div: Types[OperatorModule]['__truediv__'],
    ast.Eq: Types[OperatorModule]['__eq__'],
    ast.FloorDiv: Types[OperatorModule]['__floordiv__'],
    ast.Gt: Types[OperatorModule]['__gt__'],
    ast.GtE: Types[OperatorModule]['__ge__'],
    ast.Invert: Types[OperatorModule]['__invert__'],
    ast.Is: IsOperator,
    ast.IsNot: IsNotOperator,
    ast.Lt: Types[OperatorModule]['__lt__'],
    ast.LtE: Types[OperatorModule]['__le__'],
    ast.MatMult: Types[OperatorModule]['__matmul__'],
    ast.Mod: Types[OperatorModule]['__mod__'],
    ast.Mult: Types[OperatorModule]['__mul__'],
    ast.Not: Types[OperatorModule]['__not__'],
    ast.NotEq: Types[OperatorModule]['__ne__'],
    ast.Pow: Types[OperatorModule]['__pow__'],
    ast.Sub: Types[OperatorModule]['__sub__'],
    ast.UAdd: Types[OperatorModule]['__pos__'],
    ast.USub: Types[OperatorModule]['__neg__'],
}

Builtins = {
    'repr': {pentypes.builtins.repr.repr_},
    'slice': {pentypes.builtins.slice.slice_},
}

def astype(ty):
    return type(ty.__args__[0]) if issubclass(ty, Cst) else ty

def normalize_test_ty(types):
    return {Cst[bool(ty.__args__[0])] if issubclass(ty, Cst) else ty
            for ty in types}

def iterator_value_ty(types):
    result_types = set()
    iter_types = set()
    for ty in types:
        iter_types.update(Types[ty]['__iter__'](types))
    for ty in iter_types:
        result_types.update(Types[ty]['__next__'](iter_types))
    return result_types


class Typer(ast.NodeVisitor):

    def __init__(self, env=None):
        self.state = {}
        self.bindings = [Builtins.copy()]
        if env:
            self.bindings[0].update(env)
        self.callstack = []

    def _call(self, func, *args):
        result_type = func(*args)
        return result_type

    # expr
    def visit_BoolOp(self, node):
        operands_ty = [self.visit(value) for value in node.values]
        return set.union(*operands_ty)

    def visit_BinOp(self, node):
        operands_ty = self.visit(node.left), self.visit(node.right)
        return self._call(Ops[type(node.op)], *operands_ty)

    def visit_UnaryOp(self, node):
        operand_ty = self.visit(node.operand)
        return self._call(Ops[type(node.op)], operand_ty)

    def visit_Lambda(self, node):
        return {Lambda(node, self)}

    def visit_IfExp(self, node):
        test_ty = self.visit(node.test)
        test_ty = normalize_test_ty(test_ty)

        result_types = set()
        if test_ty != {Cst[False]}:
            result_types.update(self.visit(node.body))
        if test_ty != {Cst[True]}:
            result_types.update(self.visit(node.orelse))
        return result_types

    def visit_Dict(self, node):
        if node.keys:
            result_types = set()
            for k, v in zip(node.keys, node.values):
                dict_types = {typing.Dict[kty, vty] for kty, vty in
                              itertools.product(self.visit(k), self.visit(v))}
                result_types.update(dict_types)
            return result_types
        else:
            return {dict}

    def visit_Set(self, node):
        result_types = set()
        for e in node.elts:
            set_types = {typing.Set[ty] for ty in self.visit(e)}
            result_types.update(set_types)
        return result_types

    def visit_ListComp(self, node):
        new_bindings = {}
        no_list = False
        for generator in node.generators:
            test_types = set()
            for if_ in generator.ifs:
                test_types.update(self.visit(if_))
            test_types = normalize_test_ty(test_types)
            if test_types == {Cst[False]}:
                no_list = True

            iter_types = self.visit(generator.iter)
            if isinstance(generator.target, ast.Name):
                new_bindings[generator.target.id] = iterator_value_ty(iter_types)

        self.bindings.append(new_bindings)
        result_types = set()
        elt_types = self.visit(node.elt)
        self.bindings.pop()
        if no_list:
            return {list}
        else:
            return {typing.List[astype(elt_ty)] for elt_ty in elt_types}

    def visit_SetComp(self, node):
        new_bindings = {}
        no_set = False
        for generator in node.generators:
            test_types = set()
            for if_ in generator.ifs:
                test_types.update(self.visit(if_))
            test_types = normalize_test_ty(test_types)
            if test_types == {Cst[False]}:
                no_set = True

            iter_types = self.visit(generator.iter)
            if isinstance(generator.target, ast.Name):
                new_bindings[generator.target.id] = iterator_value_ty(iter_types)

        self.bindings.append(new_bindings)
        result_types = set()
        elt_types = self.visit(node.elt)
        self.bindings.pop()
        if no_set:
            return {set}
        else:
            return {typing.Set[astype(elt_ty)] for elt_ty in elt_types}

    def visit_DictComp(self, node):
        new_bindings = {}
        no_dict = False
        for generator in node.generators:
            test_types = set()
            for if_ in generator.ifs:
                test_types.update(self.visit(if_))
            test_types = normalize_test_ty(test_types)
            if test_types == {Cst[False]}:
                no_dict = True

            iter_types = self.visit(generator.iter)
            if isinstance(generator.target, ast.Name):
                new_bindings[generator.target.id] = iterator_value_ty(iter_types)

        self.bindings.append(new_bindings)
        result_types = set()
        key_types = self.visit(node.key)
        value_types = self.visit(node.value)
        self.bindings.pop()
        if no_dict:
            return {dict}
        else:
            return {typing.Dict[astype(k), astype(v)]
                    for k in key_types
                    for v in value_types}

    def visit_GeneratorExp(self, node):
        new_bindings = {}
        no_gen = False
        for generator in node.generators:
            test_types = set()
            for if_ in generator.ifs:
                test_types.update(self.visit(if_))
            test_types = normalize_test_ty(test_types)
            if test_types == {Cst[False]}:
                no_gen = True

            iter_types = self.visit(generator.iter)
            if isinstance(generator.target, ast.Name):
                new_bindings[generator.target.id] = iterator_value_ty(iter_types)

        self.bindings.append(new_bindings)
        result_types = set()
        elt_types = self.visit(node.elt)
        self.bindings.pop()
        if no_gen:
            return {typing.Generator}
        else:
            return {typing.Generator[astype(elt_ty), None, None] for elt_ty in elt_types}

    def visit_Compare(self, node):
        prev_ty = left_ty = self.visit(node.left)
        is_false = False
        result_types = set()
        for op, comparator in zip(node.ops, node.comparators):
            comparator_ty = self.visit(comparator)
            cmp_ty = self._call(Ops[type(op)], prev_ty, comparator_ty)
            is_false |= cmp_ty == {Cst[False]}
            result_types.update(cmp_ty)
            prev_ty = comparator_ty

        if is_false:
            return {Cst[False]}
        elif result_types == {Cst[True]}:
            return result_types
        else:
            return set(map(astype, result_types))

    def visit_Call(self, node):
        args_ty = [self.visit(arg) for arg in node.args]
        func_ty = self.visit(node.func)
        return_ty = set()
        return_ty.update(*[self._call(fty, *args_ty) for fty in func_ty])
        return return_ty

    def visit_Repr(self, node):
        value_types = self.visit(node.value)
        assert set(map(astype, value_types)) == {str}
        return {str}

    def visit_Constant(self, node):
        return {Cst[node.value]}

    def _bindattr(self, value_ty, attr):
        return lambda *args: Types[value_ty][attr]({value_ty}, *args)

    def visit_Attribute(self, node):
        value_types = self.visit(node.value)
        result_types = set()
        for value_ty in value_types:
            result_types.add(self._bindattr(value_ty, node.attr))
        return result_types

    def visit_Subscript(self, node):
        value_types = self.visit(node.value)
        slice_types = self.visit(node.slice)

        return self._call(Types[OperatorModule]['__getitem__'],
                          value_types,
                          slice_types)

    def visit_List(self, node):
        if not node.elts:
            return {list}
        result_types = set()
        for elt in node.elts:
            elt_types = self.visit(elt)
            result_types.update(map(astype, elt_types))
        return {typing.List[ty] for ty in result_types}

    def visit_Tuple(self, node):
        if not node.elts:
            return {tuple}
        result_types = []
        for elt in node.elts:
            elt_types = self.visit(elt)
            result_types.append(elt_types)
        return {tuple(tys) for tys in itertools.product(*result_types)}

    def visit_Slice(self, node):
        if node.lower:
            lower_types = self.visit(node.lower)
        else:
            lower_types = {Cst[None]}
        if node.upper:
            upper_types = self.visit(node.upper)
        else:
            upper_types = {Cst[None]}
        if node.step:
            step_types = self.visit(node.step)
        else:
            step_types = {Cst[None]}
        return next(iter(Builtins['slice']))(lower_types, upper_types, step_types)

    def visit_Name(self, node):
        for binding in reversed(self.bindings):
            if node.id in binding:
                return binding[node.id]
        raise UnboundIdentifier(node.id)

def type_eval(expr, env):
    '''
    Returns the type set of `expr` using the environment defined in `env`
    '''
    expr_node = ast.parse(expr, mode='eval')
    typer = Typer(env)
    expr_ty = typer.visit(expr_node.body)
    return expr_ty
