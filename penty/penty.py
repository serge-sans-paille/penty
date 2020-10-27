import gast as ast
import typing
import itertools
import operator
from functools import reduce

import penty.pentypes as pentypes
from penty.types import Cst, FDef, Module, astype, Lambda, Type, FilteringBool


class UnboundIdentifier(RuntimeError):
    pass


class TypeRegistry(object):

    def __init__(self):
        self.registry = {}

    def __getattr__(self, attr):
        return self.registry.attr

    def __setitem__(self, key, value):
        self.registry[key] = value

    def __contains__(self, key):
        return key in self.registry

    def __getitem__(self, key):
        if issubclass(key, Cst):
            return self.registry[type(key.__args__[0])]
        if issubclass(key, Type):
            return self.__getitem__(key.__args__[0])
        if getattr(key, '__args__', None) is not None:
            self.registry[key] = self.instanciate(key)
        return self.registry[key]

    def instanciate(self, ty):
        if ty in self.registry:
            return self.registry[ty]
        base = ty.__bases__[0]
        return self.registry[base](ty)



Types = TypeRegistry()

pentypes.builtins.register(Types)
# needed to define Ops in terms of calls to the operator module
pentypes.operator.register(Types)

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
    ast.Is: pentypes.operator.IsOperator,
    ast.IsNot: pentypes.operator.IsNotOperator,
    ast.Lt: Types[Module['operator']]['__lt__'],
    ast.LShift: Types[Module['operator']]['__lshift__'],
    ast.LtE: Types[Module['operator']]['__le__'],
    ast.MatMult: Types[Module['operator']]['__matmul__'],
    ast.Mod: Types[Module['operator']]['__mod__'],
    ast.Mult: Types[Module['operator']]['__mul__'],
    ast.Not: Types[Module['operator']]['__not__'],
    ast.NotEq: Types[Module['operator']]['__ne__'],
    ast.Pow: Types[Module['operator']]['__pow__'],
    ast.RShift: Types[Module['operator']]['__rshift__'],
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


def normalize_test_ty(types):
    if isinstance(types, set):
        return {normalize_test_ty(ty) for ty in types}
    elif issubclass(types, Cst):
        return Cst[bool(types.__args__[0])]
    else:
        return types

def is_isnone(op, left, left_ty, right, right_ty):
    if not isinstance(op, ast.Is):
        return False
    if isinstance(left, ast.Name) and right_ty == {Cst[None]}:
        return True
    if isinstance(right, ast.Name) and left_ty == {Cst[None]}:
        return True
    return False

class Typer(ast.NodeVisitor):

    def __init__(self, env=None):
        self.state = {}
        self.bindings = [Types[Module['builtins']].copy()]
        if env:
            self.bindings[0].update(env)
        self.callstack = []

    def assertValid(self):
        assert len(self.bindings) == 1, "symmetric binding pop/append"
        assert not self.callstack, "symmetric callstack pop/append"

    def _call(self, func, *args):
        # Just to make it easier to call operators
        if isinstance(func, set):
            func, = func
        if issubclass(func, FDef):
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
        elif issubclass(func, Lambda):
            lnode, = func.__args__
            new_bindings = {arg.id: arg_ty for arg, arg_ty in
                            zip(lnode.args.args, args)}
            self.bindings.append(new_bindings)
            result_type = self.visit(lnode.body)
            self.bindings.pop()
        elif issubclass(func, Type):
            result_type = self._call(Types[func.__args__[0]]['__init__'], *args)
        else:
            result_type = set()
            all_args = itertools.product(*args) if args else [[]]
            for args_ty in list(all_args):
                rty = func(*args_ty)
                if isinstance(rty, set):
                    result_type.update(rty)
                elif isinstance(rty, tuple):
                    rty, updated_tys = rty[0], rty[1]
                    result_type.add(rty)
                    for arg_ty, old_arg_ty, new_arg_ty in zip(args, args_ty, updated_tys):
                        # This is a type refinement
                        try:
                            if issubclass(new_arg_ty, old_arg_ty):
                                arg_ty.remove(old_arg_ty)
                        except TypeError:
                            pass  # happens when old_arg_ty is a parametric type
                        arg_ty.add(new_arg_ty)
                else:
                    result_type.add(rty)
        return result_type

    def _iterator_value_ty(self, types):
        iter_types = set()
        iter_types.update(*[self._call(Types[ty]['__iter__'], {ty})
                            for ty in types])
        value_types = set()
        value_types.update(*[self._call(Types[ty]['__next__'], {ty})
                            for ty in iter_types])
        return value_types

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
        value_types = self._iterator_value_ty(iter_types)
        self._type_destructuring_assign(node.target, value_types)

        return self.visit_Loop(node)

    def visit_While(self, node):
        test_types = self.visit(node.test)

        body_bindings, orelse_bindings = {}, {}
        for test_ty in test_types:
            ntest_ty = normalize_test_ty(test_ty)
            if ntest_ty is Cst[True]:
                for k, v in FilteringBool.bindings(test_ty).items():
                    body_bindings.setdefault(k, set()).update(v)
            elif ntest_ty is Cst[False]:
                for k, v in FilteringBool.bindings(test_ty).items():
                    orelse_bindings.setdefault(k, set()).update(v)

        test_types = normalize_test_ty(test_types)
        is_trivial_true = test_types == {Cst[True]}
        is_trivial_false = test_types == {Cst[False]}

        if is_trivial_false:
            for k, v in orelse_bindings.items():
                self.bindings[-1][k] = v
            return node,

        if is_trivial_true:
            self.bindings.append(body_bindings)
            for stmt in node.body:
                prev = self.visit(stmt)
                if not prev :
                    break
                if all(isinstance(p, (ast.Break, ast.Continue)) for p in prev):
                    break

            updated_bindings = self.bindings.pop()
            if not prev:
                for k, v in updated_bindings.items():
                    self.bindings[-1][k] = v
                return ()

            if all(isinstance(p, ast.Break) for p in prev):
                for stmt in node.orelse:
                    prev = self.visit(stmt)
                    if not prev:
                        break
                for k, v in updated_bindings.items():
                    self.bindings[-1][k] = v
                return prev
            elif all(not isinstance(p, ast.Break) for p in prev):
                test_types = self.visit(node.test)
                test_types = normalize_test_ty(test_types)
                is_trivial_true = test_types == {Cst[True]}

                # infinite loop detected
                if is_trivial_true:
                    self.visit_Loop(node)
                    for k, v in updated_bindings.items():
                        self.bindings[-1][k] = v
                    return ()
            for k, v in updated_bindings.items():
                self.bindings[-1][k] = v

        # other cases default to normal loop handling
        self.bindings.append(body_bindings)
        rets = self.visit_Loop(node)
        updated_bindings = self.bindings.pop()
        for k, v in updated_bindings.items():
            self.bindings[-1].setdefault(k, set()).update(v)
        for k, v in orelse_bindings.items():
            self.bindings[-1].setdefault(k, set()).update(v)
        return rets


    def visit_If(self, node):
        test_types = self.visit(node.test)

        body_bindings, orelse_bindings = {}, {}
        for test_ty in test_types:
            ntest_ty = normalize_test_ty(test_ty)
            if ntest_ty is Cst[True]:
                for k, v in FilteringBool.bindings(test_ty).items():
                    body_bindings.setdefault(k, set()).update(v)
            elif ntest_ty is Cst[False]:
                for k, v in FilteringBool.bindings(test_ty).items():
                    orelse_bindings.setdefault(k, set()).update(v)

        test_types = normalize_test_ty(test_types)

        is_trivial_true = test_types == {Cst[True]}
        is_trivial_false = test_types == {Cst[False]}

        if is_trivial_true:
            self.bindings.append(body_bindings)
            for stmt in node.body:
                prev = self.visit(stmt)
                if not prev:
                    break
                if all(isinstance(p, (ast.Break, ast.Continue)) for p in prev):
                    break
            updated_bindings = self.bindings.pop()
            for k, v in updated_bindings.items():
                self.bindings[-1][k] = v
            return prev

        if is_trivial_false:
            self.bindings.append(orelse_bindings)
            prev = node,
            for stmt in node.orelse:
                prev = self.visit(stmt)
                if not prev:
                    break
                if all(isinstance(p, (ast.Break, ast.Continue)) for p in prev):
                    break
            updated_bindings = self.bindings.pop()
            for k, v in updated_bindings.items():
                self.bindings[-1][k] = v
            return prev

        prev_body = ()
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
            packages = alias.name.split('.')
            accumulated_packages = []
            for pkg in packages[0: -1]:
                accumulated_packages.append(pkg)
                accumulated_name = ".".join(accumulated_packages)
                module = Module[accumulated_name]
                self.bindings[-1][accumulated_name] = {module}
                reduce(getattr, accumulated_packages, pentypes).register(Types)
            reduce(getattr, packages, pentypes).register(Types)
            module = Module[alias.name]
            self.bindings[-1][alias.asname or alias.name] = {module}
        return node,

    def visit_ImportFrom(self, node):
        if not node.module:
            raise NotImplementedError
        if node.level:
            raise NotImplementedError

        packages = node.module.split('.')
        accumulated_packages = []
        for pkg in packages[0: -1]:
            accumulated_packages.append(pkg)
            accumulated_name = ".".join(accumulated_packages)
            module = Module[accumulated_name]
            self.bindings[-1][accumulated_name] = {module}
            reduce(getattr, accumulated_packages, pentypes).register(Types)
        reduce(getattr, packages, pentypes).register(Types)
        module = Module[node.module]

        for alias in node.names:
            attribute = Types[module][alias.name]
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

    def _eval_chain(self, test_types, values, neutral):
        if not values:
            return test_types

        value = values.pop(0)
        next_types = set()
        for test_ty in test_types:
            ntest_ty = normalize_test_ty(test_ty)
            # neutral element => continue
            if ntest_ty is Cst[neutral]:
                next_types.add(test_ty)
                continue

            # before evaluating value, update bindings
            self.bindings.append(FilteringBool.bindings(test_ty))
            value_types = self.visit(value)
            tail_types = self._eval_chain(value_types, values, neutral)
            next_types.update(tail_types)

            # absorbing element => scoop
            if ntest_ty is Cst[not neutral]:
                pass
            else:
                next_types.add(test_ty)
            self.bindings.pop()
        return next_types

    def visit_BoolOp(self, node):
        value_types = self.visit(node.values[0])
        neutral = isinstance(node.op, ast.Or)
        return self._eval_chain(value_types, node.values[1:], neutral)

    def visit_BinOp(self, node):
        operands_ty = self.visit(node.left), self.visit(node.right)
        return self._call(Ops[type(node.op)], *operands_ty)

    def visit_UnaryOp(self, node):
        operand_ty = self.visit(node.operand)
        return self._call(Ops[type(node.op)], operand_ty)

    def visit_Lambda(self, node):
        return {Lambda[node]}

    def visit_IfExp(self, node):
        test_types = self.visit(node.test)
        result_types = set()
        for test_ty in test_types:
            ntest_ty = normalize_test_ty(test_ty)
            self.bindings.append(FilteringBool.bindings(test_ty))

            if ntest_ty is Cst[True]:
                result_types.update(self.visit(node.body))
            elif ntest_ty is Cst[False]:
                result_types.update(self.visit(node.orelse))
            else:
                result_types.update(self.visit(node.body))
                result_types.update(self.visit(node.orelse))

            # Should we check that there's no change here before discarding
            self.bindings.pop()
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
                new_bindings[generator.target.id] = self._iterator_value_ty(iter_types)

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
                new_bindings[generator.target.id] = self._iterator_value_ty(iter_types)

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
                new_bindings[generator.target.id] = self._iterator_value_ty(iter_types)

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
                new_bindings[generator.target.id] = self._iterator_value_ty(iter_types)

        self.bindings.append(new_bindings)
        result_types = set()
        elt_types = self.visit(node.elt)
        self.bindings.pop()
        if no_gen:
            return {typing.Generator}
        else:
            return {typing.Generator[astype(elt_ty), None, None] for elt_ty in elt_types}

    def _handle_is(self, prev, prev_ty, comparator, comparator_ty):
        cmp_ty = set()
        for pty, cty in itertools.product(prev_ty, comparator_ty):
            rem_pty = tuple(prev_ty - {pty})
            rem_cty = tuple(comparator_ty - {cty})
            if pty is cty:
                # special handling for singleton types
                if issubclass(pty, (Cst, Module, Type)):
                    tmp_ty = set()
                    if rem_pty and isinstance(prev, ast.Name):
                        tmp_ty.add(FilteringBool[True, prev.id, (pty,)])
                    if rem_cty and isinstance(comparator, ast.Name):
                        tmp_ty.add(FilteringBool[True, comparator.id, (cty,)])
                    if tmp_ty:
                        cmp_ty.update(tmp_ty)
                    else:
                        cmp_ty.add(Cst[True])
                else:
                    cmp_ty.add(bool)
            elif astype(pty) is astype(cty):
                if issubclass(pty, (Cst, Module, Type)) and issubclass(cty, (Cst, Module, Type)):
                    cmp_ty.add(Cst[False])
                else:
                    cmp_ty.add(bool)
            else:
                tmp_ty = set()
                if rem_pty and isinstance(prev, ast.Name):
                    tmp_ty.add(FilteringBool[False, prev.id, (pty,)])
                if rem_cty and isinstance(comparator, ast.Name):
                    tmp_ty.add(FilteringBool[False, comparator.id, (cty,)])
                if tmp_ty:
                    cmp_ty.update(tmp_ty)
                else:
                    tmp_ty = self._call(Ops[ast.Is], prev_ty, comparator_ty)
                    cmp_ty.update(tmp_ty)
        return cmp_ty


    def visit_Compare(self, node):
        prev = node.left
        prev_ty = left_ty = self.visit(node.left)
        filters = set()
        for op, comparator in zip(node.ops, node.comparators):
            comparator_ty = self.visit(comparator)
            if isinstance(op, ast.Is):
                cmp_ty = self._handle_is(prev, prev_ty, comparator, comparator_ty)
            else:
                cmp_ty = self._call(Ops[type(op)], prev_ty, comparator_ty)
            if normalize_test_ty(cmp_ty) == {Cst[False]}:
                return cmp_ty
            filters.update(ty for ty in cmp_ty if issubclass(ty, FilteringBool))
            prev, prev_ty = comparator, comparator_ty

        return cmp_ty | filters

    def visit_Call(self, node):
        args_ty = [self.visit(arg) for arg in node.args]
        func_ty = self.visit(node.func)

        return_ty = set()
        istype_compat = node.args and isinstance(node.args[0], ast.Name)
        typety = Types[Module['builtins']]['type']
        for fty in func_ty:
            if func_ty is typety and istype_compat:
                return_ty.update({fty(arg_ty, node.args[0])
                                  for arg_ty in args_ty[0]})
            else:
                return_ty.update(self._call(fty, *args_ty))
        return return_ty

    def visit_Repr(self, node):
        value_types = self.visit(node.value)
        assert set(map(astype, value_types)) == {str}
        return {str}

    def visit_Constant(self, node):
        return {Cst[node.value]}

    def _bounded_attr(self, self_set, self_ty, attr):
        func = self._unbounded_attr(self_ty, attr)
        def bounded_attr_adjustment(return_tuple):
            if isinstance(return_tuple, tuple):
                return_ty, update_ty = return_tuple
                update_self_ty, adjusted_update_ty = update_ty[0], update_ty[1:]
                # This is a type refinement
                try:
                    if issubclass(update_self_ty, self_ty):
                        self_set.remove(self_ty)
                except TypeError:
                    pass  # happens when self_ty is a parametric_type
                self_set.add(update_self_ty)
                return return_ty, adjusted_update_ty
            else:
                return return_tuple
        return Cst[lambda *args: bounded_attr_adjustment(func(self_ty, *args))]

    def _unbounded_attr(self, value_ty, attr):
        return Types[value_ty][attr]

    def visit_Attribute(self, node):
        self_types = self.visit(node.value)
        result_types = set()
        for self_ty in list(self_types):
            if issubclass(self_ty, (Module, Type)):
                result_types.add(self._unbounded_attr(self_ty, node.attr))
            else:
                result_types.add(self._bounded_attr(self_types, self_ty, node.attr))
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
        return self._call(Types[Module['builtins']]['slice'], lower_types, upper_types, step_types)

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
    typer.assertValid()
    return expr_ty

def type_exec(stmt, env):
    '''
    Returns the type environment after executing `stmt`,
    using the environment defined in `env`
    '''
    stmt_node = ast.parse(stmt, mode='exec')
    typer = Typer(env)
    typer.visit(stmt_node)
    typer.assertValid()
    top_level_bindings, = typer.bindings
    return top_level_bindings
