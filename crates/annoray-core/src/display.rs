//! xarray-style `Display` for variables, data arrays and datasets.

use std::fmt::{self, Display, Write as _};

use crate::array::typed_dispatch;
use crate::dataarray::DataArrayBase;
use crate::dataset::DatasetBase;
use crate::{Attrs, DType, Element, NaT, Name, Repr, TypedArray, VariableBase};

const PREVIEW: usize = 6;

/// In-memory bytes per element of a dtype.
fn dtype_size(d: DType) -> usize {
    match d {
        DType::I8 | DType::U8 | DType::Bool => 1,
        DType::I16 | DType::U16 => 2,
        DType::I32 | DType::U32 | DType::F32 => 4,
        DType::I64 | DType::U64 | DType::F64 => 8,
        DType::Str => size_of::<String>(),
        DType::DateTime => size_of::<chrono::NaiveDateTime>(),
        DType::CfDate => size_of::<annoray_units::CfDate>(),
        DType::TimeDelta => size_of::<chrono::TimeDelta>(),
    }
}

/// Bytes the variable occupies (or will, once loaded) — never forces a read.
fn nbytes<S: Repr>(v: &VariableBase<S>) -> usize {
    dtype_size(v.dtype()).saturating_mul(v.len())
}

/// The values, or xarray's placeholder for data not read yet.
fn body<S: Repr>(v: &VariableBase<S>) -> String {
    /// Times as text, with NaT spelled out.
    fn times<T: NaT + Display, D: ndarray::Data<Elem = T>>(
        x: &ndarray::ArrayBase<D, ndarray::IxDyn>,
    ) -> String {
        let text = x.map(|t| {
            if t.is_nat() {
                "NaT".to_owned()
            } else {
                t.to_string()
            }
        });
        format!("{text}")
    }
    match v.typed() {
        Ok(t) if !v.is_lazy() => match &t {
            TypedArray::DateTime(x) => times(x),
            TypedArray::TimeDelta(x) => times(x),
            TypedArray::CfDate(x) => times(x),
            _ => typed_dispatch!(&t, x => format!("{x}")),
        },
        _ => format!("[{} values with dtype={}]", v.len(), v.dtype()),
    }
}

fn human_bytes(n: usize) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    #[allow(clippy::cast_precision_loss)]
    let mut v = n as f64;
    let mut unit = "B";
    for u in UNITS {
        unit = u;
        if v < 1000.0 {
            break;
        }
        v /= 1000.0;
    }
    if unit == "B" {
        format!("{n}B")
    } else {
        format!("{v:.1}{unit}")
    }
}

/// Up to `PREVIEW` values, flattened, with `...` if there are more.
fn preview<S: Repr>(a: &TypedArray<S>) -> String {
    fn go<T: Element, D: ndarray::Data<Elem = T>>(
        x: &ndarray::ArrayBase<D, ndarray::IxDyn>,
        quote: bool,
    ) -> String {
        let mut out = String::new();
        for (i, v) in x.iter().enumerate() {
            if i >= PREVIEW {
                out.push_str(" ...");
                break;
            }
            if i > 0 {
                out.push(' ');
            }
            let _ = if quote {
                write!(out, "{:?}", v.to_string())
            } else {
                write!(out, "{v}")
            };
        }
        out
    }
    fn times<T: Element + NaT, D: ndarray::Data<Elem = T>>(
        x: &ndarray::ArrayBase<D, ndarray::IxDyn>,
    ) -> String {
        let mut out = String::new();
        for (i, v) in x.iter().enumerate() {
            if i >= PREVIEW {
                out.push_str(" ...");
                break;
            }
            if i > 0 {
                out.push(' ');
            }
            let _ = if v.is_nat() {
                write!(out, "NaT")
            } else {
                write!(out, "{v}")
            };
        }
        out
    }
    let quote = a.dtype() == DType::Str;
    match a {
        TypedArray::DateTime(x) => times(x),
        TypedArray::TimeDelta(x) => times(x),
        TypedArray::CfDate(x) => times(x),
        _ => typed_dispatch!(a, x => go(x, quote)),
    }
}

fn dims_tuple(dims: &[Name]) -> String {
    let inner: Vec<&str> = dims.iter().map(Name::as_str).collect();
    format!("({})", inner.join(", "))
}

fn sizes_tuple<'a>(sizes: impl Iterator<Item = (&'a Name, usize)>) -> String {
    let inner: Vec<String> = sizes.map(|(d, n)| format!("{d}: {n}")).collect();
    format!("({})", inner.join(", "))
}

/// One summary line: `  * time     (time) datetime [unit] 2020-01-01 ...`.
fn var_line<S: Repr>(
    f: &mut fmt::Formatter<'_>,
    name: &Name,
    v: &VariableBase<S>,
    marker: char,
    width: usize,
) -> fmt::Result {
    write!(
        f,
        "  {marker} {:<width$} {} {}",
        name.as_str(),
        dims_tuple(v.dims()),
        v.dtype()
    )?;
    if let Some(u) = v.unit() {
        write!(f, " [{u}]")?;
    }
    if v.is_lazy() {
        return writeln!(f, " ...");
    }
    match v.typed() {
        Ok(t) => writeln!(f, " {}", preview(&t)),
        Err(e) => writeln!(f, " <{e}>"),
    }
}

fn attrs_block(f: &mut fmt::Formatter<'_>, attrs: &Attrs) -> fmt::Result {
    if attrs.is_empty() {
        return Ok(());
    }
    writeln!(f, "Attributes:")?;
    let width = attrs
        .keys()
        .map(|k| k.len())
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    for (k, v) in attrs {
        let key = format!("{k}:");
        let value = match v {
            crate::AttrValue::Str(s) => s.clone(),
            other => format!("{other:?}"),
        };
        writeln!(f, "    {key:<width$} {value}")?;
    }
    Ok(())
}

impl<S: Repr> Display for VariableBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "<annoray.Variable {}> Size: {}",
            sizes_tuple(self.sizes()),
            human_bytes(nbytes(self))
        )?;
        writeln!(f, "{}", body(self))?;
        if let Some(u) = self.unit() {
            writeln!(f, "Units: {u}")?;
        }
        attrs_block(f, self.attrs())
    }
}

impl<S: Repr> Display for DataArrayBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self
            .name()
            .map(|n| format!(" {:?}", n.as_str()))
            .unwrap_or_default();
        writeln!(
            f,
            "<annoray.DataArray{name} {}> Size: {}",
            sizes_tuple(self.sizes()),
            human_bytes(nbytes(self.variable()))
        )?;
        writeln!(f, "{}", body(self.variable()))?;
        if let Some(u) = self.unit() {
            writeln!(f, "Units: {u}")?;
        }
        if !self.coords().is_empty() {
            writeln!(f, "Coordinates:")?;
            let width = self.coords().keys().map(|k| k.len()).max().unwrap_or(0);
            for (k, c) in self.coords() {
                let marker = if c.dims() == [k.as_str()] { '*' } else { ' ' };
                var_line(f, k, c, marker, width)?;
            }
        }
        attrs_block(f, self.attrs())
    }
}

impl<S: Repr> Display for DatasetBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total = self
            .coords()
            .values()
            .map(nbytes)
            .chain(self.data_vars().values().map(nbytes))
            .fold(0usize, usize::saturating_add);
        writeln!(f, "<annoray.Dataset> Size: {}", human_bytes(total))?;
        let sizes = self.sizes();
        writeln!(
            f,
            "Dimensions:  {}",
            sizes_tuple(sizes.iter().map(|(d, n)| (d, *n)))
        )?;
        let width = self
            .coords()
            .keys()
            .chain(self.data_vars().keys())
            .map(|k| k.len())
            .max()
            .unwrap_or(0);
        if !self.coords().is_empty() {
            writeln!(f, "Coordinates:")?;
            for (k, c) in self.coords() {
                let marker = if c.dims() == [k.as_str()] { '*' } else { ' ' };
                var_line(f, k, c, marker, width)?;
            }
        }
        writeln!(f, "Data variables:")?;
        if self.data_vars().is_empty() {
            writeln!(f, "    *empty*")?;
        }
        for (k, v) in self.data_vars() {
            var_line(f, k, v, ' ', width)?;
        }
        attrs_block(f, self.attrs())
    }
}

impl<S: Repr> fmt::Debug for DataArrayBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<S: Repr> fmt::Debug for DatasetBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}
