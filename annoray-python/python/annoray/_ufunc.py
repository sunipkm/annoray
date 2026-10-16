"""`apply_ufunc`: apply a function on plain arrays to labelled data, as xarray's.

Arguments that are DataArrays are aligned (`join`, "exact" by default as xarray; ignoring
`exclude_dims`), and their non-core dimensions broadcast by name: each array reaches `func`
with the broadcast dimensions first (length 1 where it lacks one, so numpy broadcasting
applies) and its core dimensions last, in `input_core_dims` order. Datasets apply `func` per
data variable (`dataset_join`). Other arguments (numbers, numpy arrays) are passed as they are.

The result has the broadcast dimensions followed by `output_core_dims`, the coordinates of
the inputs that fit, and the name the inputs share. `func` sees plain numbers (units are not
passed); `output_units` sets each output's unit (annoray's extension). With `keep_attrs`, the
first argument's attributes, and its unit unless `output_units` says otherwise, are kept, as
xarray keeps a `units` attribute. Computation is eager: the `dask` options are accepted and
ignored.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from annoray._annoray import DataArray, Dataset, Unit, align


def _names(spec: Sequence[Sequence[Hashable]] | None, n: int, what: str) -> list[tuple[str, ...]]:
    if spec is None:
        return [()] * n
    out = [tuple(str(d) for d in s) for s in spec]
    if len(out) != n:
        raise ValueError(f"{what} has {len(out)} entries for {n} arguments")
    return out


def _result_name(arrays: Sequence[DataArray]) -> str | None:
    names = {a.name for a in arrays}
    return names.pop() if len(names) == 1 else None


def _apply_arrays(
    func: Callable[..., Any],
    args: Sequence[Any],
    core: list[tuple[str, ...]],
    out_core: list[tuple[str, ...]],
    exclude_dims: frozenset[str],
    join: str,
    keep_attrs: bool,
    kwargs: Mapping[str, Any],
    output_units: Sequence[Any] | None,
    on_missing_core_dim: str,
) -> Any:
    positions = [i for i, a in enumerate(args) if isinstance(a, DataArray)]
    arrays = [args[i] for i in positions]
    for i in positions:
        missing = [d for d in core[i] if d not in args[i].dims]
        if missing:
            if on_missing_core_dim == "raise":
                raise ValueError(f"argument {i} lacks core dimension(s) {missing}")
            raise NotImplementedError("on_missing_core_dim other than 'raise'")
    # Align, leaving excluded dimensions free to differ (renamed apart for the alignment).
    if len(arrays) > 1:
        renamed = []
        for k, a in enumerate(arrays):
            mapping = {d: f"__excluded_{k}_{d}" for d in exclude_dims if d in a.dims}
            renamed.append(a.rename(mapping) if mapping else a)
        aligned = list(align(*renamed, join=join))  # type: ignore[arg-type]
        arrays = []
        for k, a in enumerate(aligned):
            back = {
                f"__excluded_{k}_{d}": d for d in exclude_dims if f"__excluded_{k}_{d}" in a.dims
            }
            arrays.append(a.rename(back) if back else a)
    by_position = dict(zip(positions, arrays, strict=True))

    all_core = {d for c in core for d in c}
    broadcast: list[str] = []
    sizes: dict[str, int] = {}
    for i in positions:
        a = by_position[i]
        for d, n in zip(a.dims, a.shape, strict=True):
            if d in core[i]:
                continue
            if d in all_core:
                raise ValueError(f"{d!r} is a core dimension of one argument but not another")
            if d not in sizes:
                broadcast.append(d)
                sizes[d] = n
            elif sizes[d] != n:
                raise ValueError(f"dimension {d!r} has lengths {sizes[d]} and {n}")

    def prepared(i: int) -> Any:
        a = by_position[i]
        present = [d for d in broadcast if d in a.dims]
        data = np.asarray(a.transpose(*present, *core[i]).values)
        shape = [sizes[d] if d in a.dims else 1 for d in broadcast]
        shape += [a.sizes[d] for d in core[i]]
        return data.reshape(shape)

    call_args = [prepared(i) if i in by_position else a for i, a in enumerate(args)]
    result = func(*call_args, **kwargs)
    results = result if len(out_core) > 1 else (result,)
    if len(out_core) > 1 and (not isinstance(result, tuple) or len(result) != len(out_core)):
        raise ValueError(f"func returned {type(result).__name__}, expected {len(out_core)} outputs")
    units = list(output_units) if output_units is not None else [None] * len(out_core)
    if len(units) != len(out_core):
        raise ValueError("output_units needs one unit per output")

    first = arrays[0] if arrays else None
    name = _result_name(arrays)
    outs = []
    for r, oc, unit in zip(results, out_core, units, strict=True):
        r = np.asarray(r)
        dims = (*broadcast, *oc)
        if r.ndim != len(dims):
            raise ValueError(f"func returned {r.ndim} dimensions; expected {len(dims)} {dims}")
        for d, n in zip(broadcast, r.shape, strict=False):
            if n != sizes[d]:
                raise ValueError(f"func changed the length of broadcast dimension {d!r}")
        attrs: dict[str, Any] = dict(first.attrs) if (keep_attrs and first is not None) else {}
        if unit is not None:
            attrs["units"] = str(unit) if isinstance(unit, Unit) else unit
        elif keep_attrs and first is not None and first.unit is not None:
            attrs["units"] = str(first.unit)
        out = DataArray(r, dims=dims, attrs=attrs, name=name)
        out_sizes = dict(zip(dims, r.shape, strict=True))
        coords: dict[str, Any] = {}
        for a in arrays:
            for k, c in a.coords.items():
                if k in coords:
                    continue
                if all(
                    d in out_sizes and out_sizes[d] == n
                    for d, n in zip(c.dims, c.shape, strict=True)
                ):
                    coords[k] = c
        if coords:
            out = out.assign_coords(coords)
        outs.append(out)
    return tuple(outs) if len(out_core) > 1 else outs[0]


def apply_ufunc(
    func: Callable[..., Any],
    *args: Any,
    input_core_dims: Sequence[Sequence[Hashable]] | None = None,
    output_core_dims: Sequence[Sequence[Hashable]] | None = ((),),
    exclude_dims: frozenset[Hashable] | set[Hashable] = frozenset(),
    vectorize: bool = False,
    join: str = "exact",
    dataset_join: str = "exact",
    dataset_fill_value: object = None,
    keep_attrs: bool | str | None = None,
    kwargs: Mapping[str, Any] | None = None,
    dask: str = "forbidden",
    output_dtypes: Sequence[Any] | None = None,
    output_sizes: Mapping[Hashable, int] | None = None,
    meta: Any = None,
    dask_gufunc_kwargs: Mapping[str, Any] | None = None,
    on_missing_core_dim: str = "raise",
    output_units: Sequence[Unit | str | None] | None = None,
) -> Any:
    """Apply `func` (which works on plain arrays) to DataArrays and Datasets; see the module
    notes. `output_units` (annoray) gives each output's unit."""
    del dask, output_dtypes, output_sizes, meta, dask_gufunc_kwargs, dataset_fill_value
    core = _names(input_core_dims, len(args), "input_core_dims")
    out_core = _names(output_core_dims, len(output_core_dims or ((),)), "output_core_dims")
    excluded = frozenset(str(d) for d in exclude_dims)
    for d in excluded:
        if not any(d in c for c in core):
            raise ValueError(f"excluded dimension {d!r} must be a core dimension")
    keep = keep_attrs in (True, "override", "drop_conflicts", "identical", "no_conflicts")
    kw = dict(kwargs or {})
    if vectorize:
        sig = ",".join("(" + ",".join(c) + ")" for c in core) + "->"
        sig += ",".join("(" + ",".join(c) + ")" for c in out_core)
        func = np.vectorize(func, signature=sig, excluded=set(kw))

    if any(isinstance(a, Dataset) for a in args):
        datasets = [a for a in args if isinstance(a, Dataset)]
        names = [list(ds.data_vars) for ds in datasets]
        if dataset_join == "exact":
            if any(set(n) != set(names[0]) for n in names):
                raise ValueError("data variables differ between datasets (dataset_join='exact')")
            chosen = names[0]
        elif dataset_join == "inner":
            chosen = [k for k in names[0] if all(k in n for n in names)]
        elif dataset_join == "left":
            chosen = names[0]
        elif dataset_join == "right":
            chosen = names[-1]
        else:
            raise NotImplementedError(f"dataset_join={dataset_join!r}")
        per_output: list[dict[str, DataArray]] = [{} for _ in out_core]
        for k in chosen:
            sub = [a[k] if isinstance(a, Dataset) else a for a in args]
            got = _apply_arrays(
                func,
                sub,
                core,
                out_core,
                excluded,
                join,
                keep,
                kw,
                output_units,
                on_missing_core_dim,
            )
            for slot, g in zip(per_output, got if len(out_core) > 1 else (got,), strict=True):
                slot[k] = g
        first = datasets[0]
        made = [Dataset(d, attrs=dict(first.attrs) if keep else None) for d in per_output]
        return tuple(made) if len(out_core) > 1 else made[0]

    if not any(isinstance(a, DataArray) for a in args):
        return func(*args, **kw)
    return _apply_arrays(
        func, args, core, out_core, excluded, join, keep, kw, output_units, on_missing_core_dim
    )
