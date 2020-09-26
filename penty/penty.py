import gast as ast
import typing
import itertools

import penty.pentypes as pentypes


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
        if hasattr(key, 'mro'):
            return self.registry[key]
        else:
            return self.registry[type(key)]



Types = TypeRegistry({
    bool : {
        '__not__': pentypes.bool.not_,
    },
    int: {
        '__add__': pentypes.int.add,
        '__and__': pentypes.int.bitand,
        '__bool__': pentypes.int.boolean,
        '__floordiv__': pentypes.int.floordiv,
        '__invert__': pentypes.int.invert,
        '__mul__': pentypes.int.mul,
        '__mod__': pentypes.int.mod,
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
        '__floordiv__': BinaryOperator('__floordiv__'),
        '__invert__': UnaryOperator('__invert__'),
        '__matmul__': BinaryOperator('__matmul__'),
        '__mod__': BinaryOperator('__mod__'),
        '__mul__': BinaryOperator('__mul__'),
        '__neg__': UnaryOperator('__neg__'),
        '__not__': NotOperator(),
        '__or__': BinaryOperator('__or__'),
        '__pos__': UnaryOperator('__pos__'),
        '__pow__': BinaryOperator('__pow__'),
        '__sub__': BinaryOperator('__sub__'),
        '__truediv__': BinaryOperator('__truediv__'),
        '__xor__': BinaryOperator('__xor__'),
    },
})

Ops = {
    ast.Add: Types[OperatorModule]['__add__'],
    ast.BitAnd: Types[OperatorModule]['__and__'],
    ast.BitOr: Types[OperatorModule]['__or__'],
    ast.BitXor: Types[OperatorModule]['__xor__'],
    ast.Div: Types[OperatorModule]['__truediv__'],
    ast.FloorDiv: Types[OperatorModule]['__floordiv__'],
    ast.Invert: Types[OperatorModule]['__invert__'],
    ast.MatMult: Types[OperatorModule]['__matmul__'],
    ast.Mod: Types[OperatorModule]['__mod__'],
    ast.Mult: Types[OperatorModule]['__mul__'],
    ast.Not: Types[OperatorModule]['__not__'],
    ast.Pow: Types[OperatorModule]['__pow__'],
    ast.Sub: Types[OperatorModule]['__sub__'],
    ast.UAdd: Types[OperatorModule]['__pos__'],
    ast.USub: Types[OperatorModule]['__neg__'],
}

def astype(ty):
    return ty if hasattr(ty, 'mro') else type(ty)

def normalize_test_ty(types):
    return {ty if hasattr(ty, 'mro') else bool(ty) for ty in types}

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
        self.bindings = [{} if env is None else env]
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
        if test_ty != {False}:
            result_types.update(self.visit(node.body))
        if test_ty != {True}:
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
            if test_types == {False}:
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
            if test_types == {False}:
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
            if test_types == {False}:
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
            if test_types == {False}:
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

    def visit_Call(self, node):
        args_ty = [self.visit(arg) for arg in node.args]
        func_ty = self.visit(node.func)
        return_ty = set()
        return_ty.update(*[self._call(fty, *args_ty) for fty in func_ty])
        return return_ty


    def visit_Constant(self, node):
        return {node.value}

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
