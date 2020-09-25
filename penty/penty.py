import gast as ast

import penty.pentypes as pentypes

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

Types = {
    int: {
        '__add__': pentypes.int.add,
        '__and__': pentypes.int.bitand,
        '__floordiv__': pentypes.int.floordiv,
        '__invert__': pentypes.int.invert,
        '__mul__': pentypes.int.mul,
        '__mod__': pentypes.int.mod,
        '__neg__': pentypes.int.neg,
        '__or__': pentypes.int.bitor,
        '__pos__': pentypes.int.pos,
        '__pow__': pentypes.int.pow,
        '__sub__': pentypes.int.sub,
        '__truediv__': pentypes.int.truediv,
        '__xor__': pentypes.int.bitxor,
    },
    'operator': {
        '__add__': BinaryOperator('__add__'),
        '__and__': BinaryOperator('__and__'),
        '__floordiv__': BinaryOperator('__floordiv__'),
        '__invert__': UnaryOperator('__invert__'),
        '__matmul__': BinaryOperator('__matmul__'),
        '__mod__': BinaryOperator('__mod__'),
        '__mul__': BinaryOperator('__mul__'),
        '__neg__': UnaryOperator('__neg__'),
        '__or__': BinaryOperator('__or__'),
        '__pos__': UnaryOperator('__pos__'),
        '__pow__': BinaryOperator('__pow__'),
        '__sub__': BinaryOperator('__sub__'),
        '__truediv__': BinaryOperator('__truediv__'),
        '__xor__': BinaryOperator('__xor__'),
    },
}

Ops = {
    ast.Add: Types['operator']['__add__'],
    ast.BitAnd: Types['operator']['__and__'],
    ast.BitOr: Types['operator']['__or__'],
    ast.BitXor: Types['operator']['__xor__'],
    ast.Div: Types['operator']['__truediv__'],
    ast.FloorDiv: Types['operator']['__floordiv__'],
    ast.Invert: Types['operator']['__invert__'],
    ast.MatMult: Types['operator']['__matmul__'],
    ast.Mod: Types['operator']['__mod__'],
    ast.Mult: Types['operator']['__mul__'],
    ast.Pow: Types['operator']['__pow__'],
    ast.Sub: Types['operator']['__sub__'],
    ast.UAdd: Types['operator']['__pos__'],
    ast.USub: Types['operator']['__neg__'],
}

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

    def visit_Constant(self, node):
        return {type(node.value)}

    def visit_UnaryOp(self, node):
        operand_ty = self.visit(node.operand)
        return self._call(Ops[type(node.op)], operand_ty)

    def visit_BinOp(self, node):
        operands_ty = self.visit(node.left), self.visit(node.right)
        return self._call(Ops[type(node.op)], *operands_ty)

def type_eval(expr, env):
    '''
    Returns the type set of `expr` using the environment defined in `env`
    '''
    expr_node = ast.parse(expr, mode='eval')
    typer = Typer(env)
    expr_ty = typer.visit(expr_node.body)
    return expr_ty
