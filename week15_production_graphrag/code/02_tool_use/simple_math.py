"""A tiny SAFE expression evaluator for the calculator tool.

Week 3's demo used eval() with a ⚠️ warning. Here's the production-safe version
the skill points you toward: parse the expression into an AST and only allow
arithmetic nodes — no function calls, no attribute access, no names.
"""
import ast
import operator

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported or unsafe expression")


def safe_eval(expression: str):
    """Evaluate basic arithmetic safely. Returns a number or an error string."""
    try:
        return _eval(ast.parse(expression, mode="eval").body)
    except Exception as e:  # noqa: BLE001 — tools should return errors, not crash
        return f"Error: {e}"
