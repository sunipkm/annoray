"""``query``: select along dimensions by boolean expressions over variables
(xarray's ``Dataset.query`` / ``DataArray.query``): ``ds.query(x="a > 2 & lab == 'q'")``.

Expressions are evaluated by a small restricted evaluator (no ``eval``, no pandas): names
of variables and coordinates (and, for a DataArray, its own name), numbers, strings,
arithmetic, comparisons (chained ones too, and ``in``/``not in`` with a list), ``and``,
``or``, ``not``, ``&``, ``|``, ``~``, attribute access on arrays and their accessors
(``time.dt.month``), ``.str`` methods (``lab.str.startswith('p')``), and ``Q(value,
unit)`` for quantities. With the default ``parser="pandas"``, ``&`` and ``|`` bind like
``and`` and ``or`` (``a > 2 & x < 2`` is ``(a > 2) & (x < 2)``), as in pandas.

The expression is evaluated on annoray arrays, so comparisons are units-aware: a bare
number beside data with a unit is refused unless it is 0 (``temp > 280`` on data in K is a
``UnitError``; write ``temp > Q(280, 'K')``). xarray compares the stored numbers.
"""

from __future__ import annotations

import ast
import io
import operator
import tokenize
from typing import Any

import numpy as np

from annoray._annoray import DataArray, Dataset, Quantity

_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}
_CMPOPS: dict[type, Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _pandas_precedence(expr: str) -> str:
    """``&`` → ``and``, ``|`` → ``or`` (pandas' parser), so they bind below comparisons."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(expr).readline):
        if tok.type == tokenize.OP and tok.string in ("&", "|"):
            out.append((tokenize.NAME, "and" if tok.string == "&" else "or"))
        else:
            out.append((tok.type, tok.string))
    return tokenize.untokenize(out)


class _Evaluator:
    def __init__(self, names: dict[str, Any]) -> None:
        self.names = names

    def eval(self, node: ast.AST) -> Any:
        method = getattr(self, "_" + type(node).__name__, None)
        if method is None:
            raise ValueError(f"query: {type(node).__name__} is not allowed in expressions")
        return method(node)

    def _Expression(self, n: ast.Expression) -> Any:
        return self.eval(n.body)

    def _Constant(self, n: ast.Constant) -> Any:
        if not isinstance(n.value, (int, float, str, bool)):
            raise ValueError(f"query: constant {n.value!r} is not allowed")  # noqa: TRY004
        return n.value

    def _Name(self, n: ast.Name) -> Any:
        if n.id not in self.names:
            raise NameError(f"query: {n.id!r} is not a variable or coordinate")
        return self.names[n.id]

    def _List(self, n: ast.List) -> Any:
        return [self.eval(e) for e in n.elts]

    _Tuple = _List

    def _BoolOp(self, n: ast.BoolOp) -> Any:
        f = operator.and_ if isinstance(n.op, ast.And) else operator.or_
        values = [self.eval(v) for v in n.values]
        out = values[0]
        for v in values[1:]:
            out = f(out, v)
        return out

    def _UnaryOp(self, n: ast.UnaryOp) -> Any:
        v = self.eval(n.operand)
        if isinstance(n.op, (ast.Not, ast.Invert)):
            return ~v if isinstance(v, DataArray) else not v
        if isinstance(n.op, ast.USub):
            return -v
        return +v

    def _BinOp(self, n: ast.BinOp) -> Any:
        f = _BINOPS.get(type(n.op))
        if f is None:
            raise ValueError(f"query: operator {type(n.op).__name__} is not allowed")
        return f(self.eval(n.left), self.eval(n.right))

    def _Compare(self, n: ast.Compare) -> Any:
        left = self.eval(n.left)
        out = None
        for op, right_node in zip(n.ops, n.comparators, strict=True):
            right = self.eval(right_node)
            if isinstance(op, (ast.In, ast.NotIn)):
                r = left.isin(right)
                r = ~r if isinstance(op, ast.NotIn) else r
            else:
                f = _CMPOPS.get(type(op))
                if f is None:
                    raise ValueError(f"query: comparison {type(op).__name__} is not allowed")
                r = f(left, right)
            out = r if out is None else out & r
            left = right
        return out

    def _Attribute(self, n: ast.Attribute) -> Any:
        if n.attr.startswith("_"):
            raise ValueError(f"query: attribute {n.attr!r} is not allowed")
        value = self.eval(n.value)
        if isinstance(value, (int, float, str, bool, list)):
            raise ValueError(  # noqa: TRY004 (disallowed syntax, not an argument type)
                f"query: attribute {n.attr!r} of a constant is not allowed"
            )
        return getattr(value, n.attr)

    def _Call(self, n: ast.Call) -> Any:
        from annoray._strings import StringAccessor

        f = self.eval(n.func)
        # Only side-effect-free calls: quantities and string predicates/transforms.
        if f is not Quantity and not isinstance(getattr(f, "__self__", None), StringAccessor):
            raise ValueError("query: only Q(value, unit) and .str methods can be called")
        args = [self.eval(a) for a in n.args]
        kwargs = {k.arg: self.eval(k.value) for k in n.keywords if k.arg is not None}
        return f(*args, **kwargs)


def _names(obj: Any) -> dict[str, Any]:
    names: dict[str, Any] = {"Q": Quantity, "Quantity": Quantity}
    if isinstance(obj, Dataset):
        names.update(obj.coords)
        names.update(obj.data_vars)
    else:
        names.update(obj.coords)
        if obj.name is not None:
            names[obj.name] = obj
    return names


def query(
    obj: Any,
    queries: dict[str, str] | None = None,
    parser: str = "pandas",
    engine: Any = None,
    missing_dims: str = "raise",
    **queries_kwargs: str,
) -> Any:
    """Positions along each dimension where its expression holds: ``ds.query(x="a > 2")``;
    see the module notes."""
    if parser not in ("pandas", "python"):
        raise ValueError(f"parser must be 'pandas' or 'python', not {parser!r}")
    qs = {**(queries or {}), **queries_kwargs}
    names = _names(obj)
    sel = {}
    for dim, expr in qs.items():
        if dim not in obj.dims:
            if missing_dims == "raise":
                raise ValueError(f"query: {dim!r} is not a dimension")
            continue
        text = _pandas_precedence(expr) if parser == "pandas" else expr
        mask = _Evaluator(names).eval(ast.parse(text.strip(), mode="eval"))
        if not isinstance(mask, DataArray) or mask.dtype != bool:
            raise TypeError(f"query: {expr!r} does not give booleans")
        if mask.dims != (dim,):
            raise ValueError(
                f"query: {expr!r} must give booleans along {dim!r} alone, not {mask.dims}"
            )
        sel[dim] = np.flatnonzero(np.asarray(mask.values))
    return obj.isel(sel)
