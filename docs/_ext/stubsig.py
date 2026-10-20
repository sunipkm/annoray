"""Signatures from the type stubs.

Many annoray methods are implemented in Python and forwarded through the compiled classes,
whose objects then report ``(*args, **kwargs)``. The stubs (``_annoray.pyi``) carry the
public signatures, so autodoc takes signatures from there whenever the stub has the name.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import annoray

STUB = Path(annoray.__file__).with_name("_annoray.pyi")


def _signature(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, method: bool
) -> tuple[str, str]:
    args = fn.args
    if method and args.args and args.args[0].arg in ("self", "cls"):
        args = ast.arguments(
            posonlyargs=args.posonlyargs,
            args=args.args[1:],
            vararg=args.vararg,
            kwonlyargs=args.kwonlyargs,
            kw_defaults=args.kw_defaults,
            kwarg=args.kwarg,
            defaults=args.defaults,
        )
    ret = ast.unparse(fn.returns) if fn.returns is not None else ""
    return f"({ast.unparse(args)})", ret


def _load() -> dict[str, tuple[str, str]]:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    out: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(f"annoray.{node.name}", _signature(node, method=False))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    is_static = any(
                        isinstance(d, ast.Name) and d.id == "staticmethod"
                        for d in item.decorator_list
                    )
                    sig = _signature(item, method=not is_static)
                    if item.name in ("__init__", "__new__"):
                        out.setdefault(f"annoray.{node.name}", (sig[0], ""))
                    else:
                        out.setdefault(f"annoray.{node.name}.{item.name}", sig)
    return out


SIGNATURES = _load()


def process_signature(
    app: Any,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    signature: str | None,
    ret: str | None,
) -> tuple[str, str] | None:
    if what in ("attribute", "property", "data", "module"):
        return None
    found = SIGNATURES.get(name)
    if found is None:
        return None
    sig, stub_ret = found
    return sig, (stub_ret or ret or "")


def setup(app: Any) -> dict[str, Any]:
    app.connect("autodoc-process-signature", process_signature)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
