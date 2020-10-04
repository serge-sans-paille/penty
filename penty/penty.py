import gast as ast
import typing
import itertools
import operator

import penty.pentypes as pentypes
from penty.types import Cst, FDef, Module


class UnboundIdentifier(RuntimeError):
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
        for left_ty in list(left_types):
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
        '__iadd__': pentypes.int.iadd,
        '__iand__': pentypes.int.iand,
        '__ior__': pentypes.int.ior,
        '__itruediv__': pentypes.int.itruediv,
        '__ifloordiv__': pentypes.int.ifloordiv,
        '__imod__': pentypes.int.imod,
        '__imul__': pentypes.int.imul,
        '__isub__': pentypes.int.isub,
        '__ipow__': pentypes.int.ipow,
        '__ixor__': pentypes.int.ixor,
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
    Module['operator']: {
        '__add__': BinaryOperator('__add__'),
        '__and__': BinaryOperator('__and__'),
        '__eq__': BinaryOperator('__eq__'),
        '__floordiv__': BinaryOperator('__floordiv__'),
        '__ge__': BinaryOperator('__ge__'),
        '__getitem__': BinaryOperator('__getitem__'),
        '__gt__': BinaryOperator('__gt__'),
        '__iadd__': BinaryOperator('__iadd__'),
        '__iand__': BinaryOperator('__iand__'),
        '__ior__': BinaryOperator('__ior__'),
        '__ixor__': BinaryOperator('__ixor__'),
        '__itruediv__': BinaryOperator('__itruediv__'),
        '__ifloordiv__': BinaryOperator('__ifloordiv__'),
        '__imatmul__': BinaryOperator('__imatmul__'),
        '__imod__': BinaryOperator('__imod__'),
        '__imul__': BinaryOperator('__imul__'),
        '__ipow__': BinaryOperator('__ipow__'),
        '__isub__': BinaryOperator('__isub__'),
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
        'add': BinaryOperator('__add__'),
        'and_': BinaryOperator('__and__'),
        'eq': BinaryOperator('__eq__'),
        'floordiv': BinaryOperator('__floordiv__'),
        'ge': BinaryOperator('__ge__'),
        'getitem': BinaryOperator('__getitem__'),
        'gt': BinaryOperator('__gt__'),
        'iadd': BinaryOperator('__iadd__'),
        'iand': BinaryOperator('__iand__'),
        'ior': BinaryOperator('__ior__'),
        'ixor': BinaryOperator('__ixor__'),
        'itruediv': BinaryOperator('__itruediv__'),
        'ifloordiv': BinaryOperator('__ifloordiv__'),
        'imatmul': BinaryOperator('__imatmul__'),
        'imod': BinaryOperator('__imod__'),
        'imul': BinaryOperator('__imul__'),
        'ipow': BinaryOperator('__ipow__'),
        'isub': BinaryOperator('__isub__'),
        'invert': UnaryOperator('__invert__'),
        'le': BinaryOperator('__le__'),
        'lt': BinaryOperator('__lt__'),
        'matmul': BinaryOperator('__matmul__'),
        'mod': BinaryOperator('__mod__'),
        'mul': BinaryOperator('__mul__'),
        'ne': BinaryOperator('__ne__'),
        'neg': UnaryOperator('__neg__'),
        'not_': NotOperator(),
        'or_': BinaryOperator('__or__'),
        'pos': UnaryOperator('__pos__'),
        'pow': BinaryOperator('__pow__'),
        'sub': BinaryOperator('__sub__'),
        'truediv': BinaryOperator('__truediv__'),
        'xor': BinaryOperator('__xor__'),
    },
    typing.List: pentypes.list,
    typing.Tuple: pentypes.tuple,
    })

Ops = {
    ast.Add: Types[Module['operator']]['__add__'],
    ast.BitAnd: Types[Module['operator']]['__and__'],
    ast.BitOr: Types[Module['operator']]['__or__'],
    ast.BitXor: Types[Module['operator']]['__xor__'],
    ast.Div: Types[Module['operator']]['__truediv__'],
    ast.Eq: Types[Module['operator']]['__eq__'],
    ast.FloorDiv: Types[Module['operator']]['__floordiv__'],
    ast.Gt: Types[Module['operator']]['__gt__'],
    ast.GtE: Types[Module['operator']]['__ge__'],
    ast.Invert: Types[Module['operator']]['__invert__'],
    ast.Is: IsOperator,
    ast.IsNot: IsNotOperator,
    ast.Lt: Types[Module['operator']]['__lt__'],
    ast.LtE: Types[Module['operator']]['__le__'],
    ast.MatMult: Types[Module['operator']]['__matmul__'],
    ast.Mod: Types[Module['operator']]['__mod__'],
    ast.Mult: Types[Module['operator']]['__mul__'],
    ast.Not: Types[Module['operator']]['__not__'],
    ast.NotEq: Types[Module['operator']]['__ne__'],
    ast.Pow: Types[Module['operator']]['__pow__'],
    ast.Sub: Types[Module['operator']]['__sub__'],
    ast.UAdd: Types[Module['operator']]['__pos__'],
    ast.USub: Types[Module['operator']]['__neg__'],
}

IOps = {
    ast.Add: Types[Module['operator']]['__iadd__'],
    ast.BitAnd: Types[Module['operator']]['__iand__'],
    ast.BitOr: Types[Module['operator']]['__ior__'],
    ast.BitXor: Types[Module['operator']]['__ixor__'],
    ast.Div: Types[Module['operator']]['__itruediv__'],
    ast.FloorDiv: Types[Module['operator']]['__ifloordiv__'],
    ast.MatMult: Types[Module['operator']]['__imatmul__'],
    ast.Mod: Types[Module['operator']]['__imod__'],
    ast.Mult: Types[Module['operator']]['__imul__'],
    ast.Pow: Types[Module['operator']]['__ipow__'],
    ast.Sub: Types[Module['operator']]['__isub__'],
}

Builtins = {
    'id': {pentypes.builtins.id.id_},
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
        if hasattr(func, 'mro') and issubclass(func, FDef):
            fnode, = func.__args__
            call_key = func, *args
            if call_key in self.callstack:
                return {}

            new_bindings = {arg.id: arg_ty for arg, arg_ty in
                            zip(fnode.args.args, args)}
            new_bindings['@'] = set()
            self.bindings.append(new_bindings)
            self.callstack.append(call_key)
            for stmt in fnode.body:
                prev = self.visit(stmt)
                if not prev:
                    break

            if prev:
                new_bindings['@'].add(Cst[None])

            result_type = new_bindings['@']
            self.callstack.pop()
            self.bindings.pop()
        else:
            result_type = func(*args)
        return result_type

    def _type_destructuring_assign(self, node, types):
        if isinstance(node, ast.Name):
            self.bindings[-1][node.id] = types
        elif isinstance(node, ast.Tuple):
            for i, elt in enumerate(node.elts):
                elt_types = self._call(
                    Types[Module['operator']]['__getitem__'],
                    types, {Cst[i]})
                self._type_destructuring_assign(elt, elt_types)
        else:
            raise NotImplementedError


    # stmt
    def visit_Module(self, node):
        prev = ()
        for stmt in node.body:
            prev = self.visit(stmt)
            if not prev :
                break
        return prev

    def visit_FunctionDef(self, node):
        self.bindings[-1][node.name] = {FDef[node]}
        return node,

    def visit_Return(self, node):
        self.bindings[-1].setdefault('@', set())
        if node.value is None:
            self.bindings[-1]['@'].add(Cst[None])
        else:
            self.bindings[-1]['@'].update(self.visit(node.value))
        return ()

    def visit_Delete(self, node):
        self.generic_visit(node)
        return node,

    def visit_Assign(self, node):
        value_types = self.visit(node.value)
        for target in node.targets:
            self._type_destructuring_assign(target, value_types)
        return node,

    def visit_AugAssign(self, node):
        value_types = self.visit(node.value)
        target_types = self.visit(node.target)
        new_types = self._call(IOps[type(node.op)], target_types, value_types)
        self._type_destructuring_assign(node.target, new_types)
        return node,

    def visit_Loop(self, node):

        loop_bindings = {}
        self.bindings.append(loop_bindings)

        for stmt in node.body:
            prev = self.visit(stmt)
            if not prev :
                break
            if all(isinstance(p, (ast.Break, ast.Continue)) for p in prev):
                break

        reloop_bindings = {}

        if not all(isinstance(p, ast.Break) for p in prev):
            self.bindings.append(reloop_bindings)
            for stmt in node.body:
                prev = self.visit(stmt)
                if not prev:
                    break
                if all(isinstance(p, (ast.Break, ast.Continue)) for p in prev):
                    break

            self.bindings.pop()

        self.bindings.pop()

        for k, v in loop_bindings.items():
            if k not in self.bindings[-1]:
                self.bindings[-1][k] = v
            else:
                self.bindings[-1][k].update(v)

        for k, v in reloop_bindings.items():
            if k not in self.bindings[-1]:
                self.bindings[-1][k] = v
            elif not self.bindings[-1][k].issuperset(v):
                # get rid of constants that grows along the loop
                self.bindings[-1][k] = {astype(ty)
                                        for ty in self.bindings[-1][k]}

        if not prev:
            return ()


        if all(not isinstance(p, ast.Break) for p in prev):
            for stmt in node.orelse:
                prev = self.visit(stmt)
                if not prev:
                    break
        elif any(not isinstance(p, ast.Break) for p in prev):
            orelse_bindings = {}
            self.bindings.append(orelse_bindings)
            for stmt in node.orelse:
                prev = self.visit(stmt)
                if not prev:
                    break
            self.bindings.pop()
            for k, v in orelse_bindings.items():
                if k in self.bindings[-1]:
                    self.bindings[-1][k].update(v)
                else:
                    self.bindings[-1][k] = v

        return prev

    def visit_For(self, node):
        iter_types = self.visit(node.iter)
        value_types = iterator_value_ty(iter_types)
        self._type_destructuring_assign(node.target, value_types)

        return self.visit_Loop(node)

    def visit_While(self, node):
        test_types = self.visit(node.test)
        test_types = normalize_test_ty(test_types)

        is_trivial_true = test_types == {Cst[True]}
        is_trivial_false = test_types == {Cst[False]}

        if is_trivial_false:
            return node,

        if is_trivial_true:
            for stmt in node.body:
                prev = self.visit(stmt)
                if not prev :
                    break
                if all(isinstance(p, (ast.Break, ast.Continue)) for p in prev):
                    break

            if not prev:
                return ()

            if all(isinstance(p, ast.Break) for p in prev):
                for stmt in node.orelse:
                    prev = self.visit(stmt)
                    if not prev:
                        break
                return prev
            elif all(not isinstance(p, ast.Break) for p in prev):
                test_types = self.visit(node.test)
                test_types = normalize_test_ty(test_types)
                is_trivial_true = test_types == {Cst[True]}

                # infinite loop detected
                if is_trivial_true:
                    self.visit_Loop(node)
                    return ()

            # other cases default to normal loop handling

        return self.visit_Loop(node)


    def visit_If(self, node):
        test_types = self.visit(node.test)
        test_types = normalize_test_ty(test_types)

        is_trivial_true = test_types == {Cst[True]}
        is_trivial_false = test_types == {Cst[False]}

        if is_trivial_true:
            for stmt in node.body:
                prev = self.visit(stmt)
                if not prev:
                    break
                if all(isinstance(p, (ast.Break, ast.Continue)) for p in prev):
                    break
            return prev

        if is_trivial_false:
            prev = node,
            for stmt in node.orelse:
                prev = self.visit(stmt)
                if not prev:
                    break
                if all(isinstance(p, (ast.Break, ast.Continue)) for p in prev):
                    break
            return prev

        prev_body = ()
        body_bindings = {}
        self.bindings.append(body_bindings)
        try:
            for stmt in node.body:
                prev_body = self.visit(stmt)
                if not prev_body:
                    break
                if all(isinstance(p, (ast.Break, ast.Continue)) for p in
                       prev_body):
                    break
        except UnboundIdentifier as ui:
            pass
        finally:
            self.bindings.pop()

        prev_orelse = node,
        orelse_bindings = {}
        self.bindings.append(orelse_bindings)
        try:
            for stmt in node.orelse:
                prev_orelse = self.visit(stmt)
                if not prev_orelse:
                    break
                if all(isinstance(p, (ast.Break, ast.Continue)) for p in
                       prev_orelse):
                    break
        except UnboundIdentifier as ui:
            pass
        finally:
            self.bindings.pop()

        all_keys = set(body_bindings.keys()).union(orelse_bindings.keys())
        for k in all_keys:
            k_inbody, k_inorelse = k in body_bindings, k in orelse_bindings
            if k_inbody & k_inorelse:
                # don't erase the return type, add to it
                if k != '@':
                    self.bindings[-1][k].clear()
                self.bindings[-1][k].update(body_bindings[k],
                                            orelse_bindings[k])
            elif k_inbody:
                self.bindings[-1].setdefault(k, set())  # only in one branch
                self.bindings[-1][k].update(body_bindings[k])
            else:
                self.bindings[-1].setdefault(k, set())  # only in one branch
                self.bindings[-1][k].update(orelse_bindings[k])
        return prev_body + prev_orelse

    def visit_Import(self, node):
        for alias in node.names:
            if '.' in alias.name:
                raise NotImplementedError
            module = Module[alias.name]
            self.bindings[-1][alias.asname or alias.name] = {module}
        return node,

    def visit_ImportFrom(self, node):
        if not node.module:
            raise NotImplementedError
        if node.level:
            raise NotImplementedError
        module_path = node.module.split('.')
        if len(module_path) == 1:
            path = Types[Module[module_path[0]]]
        else:
            path = reduce(operator.getitem, module_path[1:],
                          Types[Module[module_path[0]]])
        for alias in node.names:
            attribute = path[alias.name]
            self.bindings[-1][alias.asname or alias.name] = {attribute}
        return node,

    def visit_Expr(self, node):
        self.generic_visit(node)
        return node,

    def visit_Pass(self, node):
        return node,

    def visit_Break(self, node):
        return node,

    def visit_Continue(self, node):
        return node,

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

    def _bounded_attr(self, value_types, value_ty, attr):
        func = self._unbounded_attr(value_ty, attr)
        return lambda *args: func(value_types, *args)

    def _unbounded_attr(self, value_ty, attr):
        return Types[value_ty][attr]

    def visit_Attribute(self, node):
        value_types = self.visit(node.value)
        result_types = set()
        for value_ty in list(value_types):
            if issubclass(value_ty, Module):
                result_types.add(self._unbounded_attr(value_ty, node.attr))
            else:
                result_types.add(self._bounded_attr(value_types, value_ty, node.attr))
        return result_types

    def visit_Subscript(self, node):
        value_types = self.visit(node.value)
        slice_types = self.visit(node.slice)

        return self._call(Types[Module['operator']]['__getitem__'],
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
        return {typing.Tuple[tys] for tys in itertools.product(*result_types)}

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
        # the store here is surprising, but it comes from augassign
        # so the identifier should already be bound
        if isinstance(node.ctx, (ast.Load, ast.Store)):
            for binding in reversed(self.bindings):
                if node.id in binding:
                    return binding[node.id]
        elif isinstance(node.ctx, ast.Del):
            for binding in reversed(self.bindings):
                if node.id in binding:
                    node_ty = binding[node.id]
                    del binding[node.id]
                    return node_ty

        raise UnboundIdentifier(node.id)

def type_eval(expr, env):
    '''
    Returns the type set of `expr`,
    using the environment defined in `env`
    '''
    expr_node = ast.parse(expr, mode='eval')
    typer = Typer(env)
    expr_ty = typer.visit(expr_node.body)
    return expr_ty

def type_exec(stmt, env):
    '''
    Returns the type environment after executing `stmt`,
    using the environment defined in `env`
    '''
    stmt_node = ast.parse(stmt, mode='exec')
    typer = Typer(env)
    typer.visit(stmt_node)
    top_level_bindings, = typer.bindings
    return top_level_bindings
