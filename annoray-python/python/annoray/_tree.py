"""xarray's DataTree API on annoray's tree nodes (``annoray._annoray.DataTree``: a shared
root and a path, so nodes taken from a tree are views of it).

Traversal is breadth first, as xarray's ``subtree``. ``map_over_datasets``, arithmetic and
the Dataset methods apply node by node to each node's dataset with its inherited
coordinates; results do not repeat inherited coordinates. A node removed from its tree (or
replaced) is no longer reachable through views taken earlier — take a ``copy()`` to keep
one (xarray keeps such a node alive as its own tree).
"""

from __future__ import annotations

import operator
import posixpath
from collections.abc import Callable, Iterator
from pathlib import PurePosixPath
from typing import Any

from annoray._annoray import DataArray, Dataset, DataTree

# ----- structure -----


def children(node: DataTree) -> dict[str, DataTree]:
    """Child name → node."""
    return dict(zip(node._child_names(), node._list_children(), strict=True))


def subtree(node: DataTree) -> list[DataTree]:
    """This node and every node below it, breadth first."""
    out, queue = [], [node]
    while queue:
        n = queue.pop(0)
        out.append(n)
        queue.extend(n._list_children())
    return out


def descendants(node: DataTree) -> list[DataTree]:
    return subtree(node)[1:]


def leaves(node: DataTree) -> list[DataTree]:
    return [n for n in subtree(node) if not n._child_names()]


def parents(node: DataTree) -> list[DataTree]:
    """The ancestors, nearest first."""
    out = []
    p = node.parent
    while p is not None:
        out.append(p)
        p = p.parent
    return out


def lineage(node: DataTree) -> list[DataTree]:
    return [node, *parents(node)]


def ancestors(node: DataTree) -> list[DataTree]:
    """The root first, this node last."""
    return lineage(node)[::-1]


def groups(node: DataTree) -> tuple[str, ...]:
    return tuple(n.path for n in subtree(node))


def level(node: DataTree) -> int:
    return len(parents(node))


def depth(node: DataTree) -> int:
    """The deepest level in the whole tree."""
    return max(level(n) for n in subtree(node.root))


def width(node: DataTree) -> int:
    """The number of nodes at this node's level, across the whole tree."""
    lv = level(node)
    return sum(1 for n in subtree(node.root) if level(n) == lv)


def is_leaf(node: DataTree) -> bool:
    return not node._child_names()


def has_data(node: DataTree) -> bool:
    ds = node.to_dataset(inherit=False)
    return bool(len(ds.data_vars) or len(ds.coords))


def has_attrs(node: DataTree) -> bool:
    return bool(node.to_dataset(inherit=False).attrs)


def is_empty(node: DataTree) -> bool:
    return not has_data(node) and is_leaf(node)


def relative_to(node: DataTree, other: DataTree) -> str:
    """This node's path relative to ``other`` (in the same tree)."""
    if not node._same_root(other):
        raise ValueError("relative_to: the nodes are not in the same tree")
    return posixpath.relpath(node.path, other.path)


def same_tree(node: DataTree, other: DataTree) -> bool:
    return node._same_root(other)


def find_common_ancestor(node: DataTree, other: DataTree) -> DataTree:
    if not node._same_root(other):
        raise ValueError("find_common_ancestor: the nodes are not in the same tree")
    common = posixpath.commonpath([node.path, other.path])
    found = node.root._node(common)
    if found is None:  # pragma: no cover (the common path of two nodes exists)
        raise KeyError(common)
    return found


def isomorphic(node: DataTree, other: DataTree) -> bool:
    """The same structure: the same child names at every level (any order)."""
    if not isinstance(other, DataTree):
        return False
    mine, theirs = children(node), children(other)
    return set(mine) == set(theirs) and all(isomorphic(mine[k], theirs[k]) for k in mine)


def _at(tree: DataTree, rel: str) -> DataTree:
    """The node at relative path ``rel`` (``"."`` for ``tree`` itself)."""
    found = tree if rel == "." else tree._node(rel)
    if found is None:
        raise KeyError(rel)
    return found


def _all_pairs(node: DataTree, other: DataTree) -> Iterator[tuple[DataTree, DataTree]]:
    for a in subtree(node):
        yield a, _at(other, relative_to(a, node))


def equals(node: DataTree, other: Any) -> bool:
    """Isomorphic, every node's own dataset ``equals``."""
    return (
        isinstance(other, DataTree)
        and isomorphic(node, other)
        and all(
            a.to_dataset(inherit=False).equals(b.to_dataset(inherit=False))
            for a, b in _all_pairs(node, other)
        )
    )


def identical(node: DataTree, other: Any) -> bool:
    """Isomorphic, every node's own dataset ``identical``, and the same name."""
    return (
        isinstance(other, DataTree)
        and node.name == other.name
        and isomorphic(node, other)
        and all(
            a.to_dataset(inherit=False).identical(b.to_dataset(inherit=False))
            for a, b in _all_pairs(node, other)
        )
    )


# ----- dataset-like access -----


def ds(node: DataTree) -> Dataset:
    """This node's dataset with the coordinates it inherits (a copy)."""
    return node.to_dataset(inherit=True)


def data_vars(node: DataTree) -> Any:
    return ds(node).data_vars


def coords(node: DataTree) -> Any:
    return ds(node).coords


def dims(node: DataTree) -> Any:
    return ds(node).dims


def sizes(node: DataTree) -> Any:
    return ds(node).sizes


def attrs(node: DataTree) -> Any:
    from annoray._mapping import AttrsDict

    return AttrsDict(node, _plain_attrs(node))


def _plain_attrs(node: DataTree) -> dict[str, Any]:
    return dict(node.to_dataset(inherit=False).attrs)


def _edit_attrs(node: DataTree, changed: dict[str, Any], removed: list[str]) -> None:
    own = node.to_dataset(inherit=False)
    for k in removed:
        del own.attrs[k]
    own.attrs.update(changed)
    node._set_dataset(own)


def set_attrs(node: DataTree, value: Any) -> None:
    own = node.to_dataset(inherit=False)
    own.attrs = dict(value)
    node._set_dataset(own)


def encoding(node: DataTree) -> Any:
    from annoray._mapping import EncodingDict

    return EncodingDict(node, _plain_encoding(node))


def _plain_encoding(node: DataTree) -> dict[str, Any]:
    return dict(node.to_dataset(inherit=False).encoding)


def _edit_encoding(node: DataTree, changed: dict[str, Any], removed: list[str]) -> None:
    own = node.to_dataset(inherit=False)
    enc = own.encoding
    for k in removed:
        del enc[k]
    enc.update(changed)
    node._set_dataset(own)


def _split(key: str) -> tuple[str, str]:
    parent, _, name = key.rstrip("/").rpartition("/")
    return parent, name


def getitem(node: DataTree, key: str) -> Any:
    """A node (``"a"``, ``"a/b"``, ``"/a"``) or a variable (``"v"``, ``"a/v"``; inherited
    coordinates too)."""
    if not isinstance(key, str):
        raise TypeError("DataTree keys are paths or variable names (strings)")
    found = node._node(key)
    if found is not None:
        return found
    parent, name = _split(key)
    owner = node._node(parent) if parent else node
    if owner is not None:
        data = owner.to_dataset(inherit=True)
        if name in data:
            return data[name]
    raise KeyError(key)


def setitem(node: DataTree, key: str, value: Any) -> None:
    """A child (a DataTree, copied; or a Dataset as a new node) or a variable."""
    parent, name = _split(key)
    owner = node._node(parent) if parent else node
    if owner is None:
        raise KeyError(f"no node {parent!r} to put {name!r} in")
    if isinstance(value, DataTree):
        owner._set_child(name, value)
    elif isinstance(value, Dataset):
        owner._set_child(name, DataTree(value))
    else:
        own = owner.to_dataset(inherit=False)
        if name in own.coords and not isinstance(value, DataArray):
            own.coords[name] = value
        else:
            own[name] = value
        owner._set_dataset(own)


def delitem(node: DataTree, key: str) -> None:
    parent, name = _split(key)
    owner = node._node(parent) if parent else node
    if owner is None:
        raise KeyError(key)
    if name in owner._child_names():
        owner._del_child(name)
        return
    own = owner.to_dataset(inherit=False)
    if name not in own:
        raise KeyError(key)
    del own[name]
    owner._set_dataset(own)


def contains(node: DataTree, key: str) -> bool:
    """A variable of this node (inherited coordinates included) or a child: not a path,
    as xarray."""
    return key in node._child_names() or key in node.to_dataset(inherit=True)


def keys(node: DataTree) -> list[str]:
    """The node's data variables, then its children (xarray's iteration order)."""
    return [*node.to_dataset(inherit=False).data_vars, *node._child_names()]


def length(node: DataTree) -> int:
    return len(keys(node))


def values(node: DataTree) -> list[Any]:
    return [getitem(node, k) for k in keys(node)]


def items(node: DataTree) -> list[tuple[str, Any]]:
    return [(k, getitem(node, k)) for k in keys(node)]


def get(node: DataTree, key: str, default: Any = None) -> Any:
    try:
        return getitem(node, key)
    except KeyError:
        return default


# ----- whole trees -----


def to_dict(node: DataTree, relative: bool = False) -> dict[str, Dataset]:
    """Path → each node's own dataset (absolute paths, or relative to this node)."""
    return {
        (relative_to(n, node) if relative else n.path): n.to_dataset(inherit=False)
        for n in subtree(node)
    }


def copy(node: DataTree, deep: bool = True) -> DataTree:
    """This node's subtree as a new tree (annoray never shares data between trees)."""
    return node._detached()


def _rebuild(node: DataTree, datasets: dict[str, Dataset]) -> DataTree:
    """A tree from relative path → dataset; a coordinate identical to one an ancestor
    defines is not repeated below it (as xarray rebuilds results)."""
    out = DataTree.from_dict(
        {"/" if k == "." else k: v for k, v in datasets.items()}, name=node.name
    )
    for n in descendants(out):
        above: dict[str, Any] = {}
        for p in parents(n):
            for k, c in p.to_dataset(inherit=False).coords.items():
                above.setdefault(k, c)
        own = n.to_dataset(inherit=False)
        repeated = [k for k, c in own.coords.items() if k in above and c.identical(above[k])]
        if repeated:
            n._set_dataset(own.drop_vars(repeated))
    return out


def map_over_datasets(
    node: DataTree, func: Callable[..., Any], *args: Any, kwargs: Any = None
) -> Any:
    """``func(dataset, *args)`` for every node (each node's dataset with its inherited
    coordinates; DataTree arguments give their corresponding node's dataset), as a tree of
    the results (a tuple of trees for tuple results)."""
    kwargs = kwargs or {}
    results: dict[str, Any] = {}
    for n in subtree(node):
        rel = relative_to(n, node)
        call_args = [_at(a, rel).to_dataset() if isinstance(a, DataTree) else a for a in args]
        results[rel] = func(n.to_dataset(), *call_args, **kwargs)
    first = next(iter(results.values()))
    if isinstance(first, tuple):
        return tuple(
            _rebuild(node, {k: r[i] for k, r in results.items()}) for i in range(len(first))
        )
    return _rebuild(node, results)


def map_over_datasets_fn(func: Callable[..., Any], *args: Any, kwargs: Any = None) -> Any:
    """xarray's module-level ``map_over_datasets``: the trees among ``args`` must be
    isomorphic; ``func`` gets each node's datasets in their place."""
    trees = [a for a in args if isinstance(a, DataTree)]
    if not trees:
        raise TypeError("map_over_datasets needs at least one DataTree")
    first = trees[0]
    for t in trees[1:]:
        if not isomorphic(first, t):
            raise ValueError("map_over_datasets: the trees are not isomorphic")
    idx = args.index(first)
    rest = args[:idx] + args[idx + 1 :]

    def call(ds_: Dataset, *others: Any, **kw: Any) -> Any:
        full = [*others[:idx], ds_, *others[idx:]]
        return func(*full, **kw)

    return map_over_datasets(first, call, *rest, kwargs=kwargs)


def group_subtrees(*trees: DataTree) -> Iterator[tuple[str, tuple[DataTree, ...]]]:
    """``(relative path, nodes)`` for every position of isomorphic trees, breadth first."""
    if not trees:
        return
    for t in trees[1:]:
        if not isomorphic(trees[0], t):
            raise ValueError("group_subtrees: the trees are not isomorphic")
    for n in subtree(trees[0]):
        rel = relative_to(n, trees[0])
        yield rel, tuple(n if i == 0 else _at(t, rel) for i, t in enumerate(trees))


def match(node: DataTree, pattern: str) -> DataTree:
    """The nodes whose path (relative to this one) matches the glob ``pattern``, as a new
    tree (ancestors not matched are empty; matched nodes keep what they inherited)."""
    picked = {
        relative_to(n, node): n.to_dataset(inherit=True)
        for n in subtree(node)
        if PurePosixPath(relative_to(n, node)).match(pattern)
    }
    return _rebuild(node, picked)


def filter(node: DataTree, filterfunc: Callable[[DataTree], bool]) -> DataTree:
    """The nodes for which ``filterfunc(node)`` holds, as a new tree (keeping what they
    inherited)."""
    picked = {
        relative_to(n, node): n.to_dataset(inherit=True) for n in subtree(node) if filterfunc(n)
    }
    return _rebuild(node, picked)


def filter_like(node: DataTree, other: DataTree) -> DataTree:
    """The nodes whose relative path also exists in ``other``."""
    paths = {relative_to(n, other) for n in subtree(other)}
    return filter(node, lambda n: relative_to(n, node) in paths)


def drop_nodes(node: DataTree, names: Any, *, errors: str = "raise") -> DataTree:
    """A copy without the named children."""
    names = [names] if isinstance(names, str) else list(names)
    missing = [n for n in names if n not in node._child_names()]
    if missing and errors == "raise":
        raise KeyError(f"no children {missing}")
    out = copy(node)
    for n in names:
        if n in out._child_names():
            out._del_child(n)
    return out


def prune(node: DataTree, drop_size_zero_vars: bool = False) -> DataTree:
    """A copy without the nodes whose subtree holds no data variables."""

    def keeps(n: DataTree) -> bool:
        for m in subtree(n):
            vars_ = m.to_dataset(inherit=False).data_vars
            if drop_size_zero_vars:
                if any(v.size > 0 for v in vars_.values()):
                    return True
            elif len(vars_):
                return True
        return False

    out = copy(node)
    for n in reversed(descendants(out)):
        parent = n.parent
        if not keeps(n) and parent is not None and n.name is not None:
            parent._del_child(n.name)
    return out


def pipe(node: DataTree, func: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(func, tuple):
        f, target = func
        return f(*args, **{target: node, **kwargs})
    return func(node, *args, **kwargs)


def assign(node: DataTree, items: Any = None, **kwargs: Any) -> DataTree:
    """A copy with variables (or children) added or replaced on this node."""
    out = copy(node)
    for k, v in {**(items or {}), **kwargs}.items():
        setitem(out, k, v)
    return out


def load(node: DataTree, **kwargs: Any) -> DataTree:
    """Read every lazily loaded value of this subtree, in place; returns this node."""
    node._load()
    return node


def compute(node: DataTree, **kwargs: Any) -> DataTree:
    """A copy with every value read into memory."""
    out = copy(node)
    out._load()
    return out


def chunk(node: DataTree, *args: Any, **kwargs: Any) -> DataTree:
    """No-op without dask."""
    return node


def repr_html(node: DataTree) -> str:
    import html

    return f"<pre>{html.escape(repr(node))}</pre>"


def reduce_pickle(node: DataTree) -> Any:
    return (_from_state, (to_dict(node, relative=True), node.name))


def _from_state(datasets: dict[str, Dataset], name: str | None) -> DataTree:
    return DataTree.from_dict({("/" if k == "." else k): v for k, v in datasets.items()}, name=name)


# ----- node-by-node computation -----


def _per_node(method: str) -> Callable[..., DataTree]:
    def apply(node: DataTree, *args: Any, **kwargs: Any) -> DataTree:
        def one(d: Dataset, *a: Any) -> Dataset:
            # Reductions leave a node without data as it is (nothing to reduce: xarray
            # keeps its coordinates); selections still apply to its coordinates.
            if method in _REDUCTIONS and not len(d.data_vars):
                return d
            try:
                return call(d, *a)
            except (ValueError, KeyError, TypeError):
                # A node with no data (only coordinates) is left as it is.
                if not len(d.data_vars):
                    return d
                raise

        def call(d: Dataset, *a: Any) -> Dataset:
            if method == "isel":
                return d.isel(*a, missing_dims="ignore", **kwargs)
            if method == "sel":
                wanted = {**(a[0] if a else {}), **kwargs}
                return d.sel({k: v for k, v in wanted.items() if k in d.dims})
            return getattr(d, method)(*a, **kwargs)

        # Tree arguments give each node the corresponding node's dataset.
        return map_over_datasets(node, one, *args)

    apply.__name__ = f"node_{method}"
    apply.__doc__ = f"``Dataset.{method}`` applied to every node."
    return apply


_REDUCTIONS = {
    "mean",
    "sum",
    "prod",
    "std",
    "var",
    "min",
    "max",
    "median",
    "count",
    "any",
    "all",
    "quantile",
    "cumsum",
    "cumprod",
}

METHODS = (
    "mean",
    "sum",
    "prod",
    "std",
    "var",
    "min",
    "max",
    "median",
    "count",
    "any",
    "all",
    "isel",
    "sel",
    "squeeze",
    "astype",
    "fillna",
    "dropna",
    "interp",
    "reindex",
    "round",
    "clip",
    "quantile",
    "diff",
    "shift",
    "roll",
    "cumsum",
    "cumprod",
    "isnull",
    "notnull",
    "where",
    "drop_vars",
    "rename_vars",
    "head",
    "tail",
    "thin",
)
# As node_<name>, so that min, max, sum, any and all keep their builtin meaning here.
for _name in METHODS:
    globals()[f"node_{_name}"] = _per_node(_name)

_BINOPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
    "pow": operator.pow,
    "and": operator.and_,
    "or": operator.or_,
    "xor": operator.xor,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
}


def binary(node: DataTree, other: Any, op: str, reflected: bool = False) -> Any:
    """``node op other`` node by node (a tree ``other`` must be isomorphic)."""
    f = _BINOPS[op]
    if isinstance(other, DataTree):
        if not isomorphic(node, other):
            raise ValueError("arithmetic between trees needs isomorphic trees")
        return map_over_datasets(node, lambda a, b: f(b, a) if reflected else f(a, b), other)
    return map_over_datasets(node, lambda a: f(other, a) if reflected else f(a, other))


def unary(node: DataTree, op: str) -> DataTree:
    f = {"neg": operator.neg, "pos": operator.pos, "abs": operator.abs, "invert": operator.invert}[
        op
    ]
    return map_over_datasets(node, f)


# ----- files -----


def open_groups(filename_or_obj: Any, **kwargs: Any) -> dict[str, Dataset]:
    """Every group of a file (or Zarr store) as a dataset, by path."""
    from annoray._annoray import open_datatree

    return to_dict(open_datatree(filename_or_obj, **kwargs))


def open_datatree(filename_or_obj: Any, *, engine: str | None = None, **kwargs: Any) -> DataTree:
    """Open every group of a netCDF-4/HDF5 file or Zarr store as a DataTree (lazily)."""
    from annoray._annoray import open_datatree as _open
    from annoray._io import _check

    _check(engine, {k: v for k, v in kwargs.items() if k in ("chunks", "cache", "lock")})
    return _open(
        filename_or_obj, **{k: v for k, v in kwargs.items() if k not in ("chunks", "cache", "lock")}
    )
