//! Procedural macros for annoray. Use them through the `annoray` facade crate.

#![deny(missing_docs)]

use proc_macro2::{Delimiter, Spacing, Span, TokenStream, TokenTree};
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Error, Expr, Ident};

mod schema;

/// A struct of typed views of a dataset: `Name::schema()` says what the dataset must hold, and
/// `Name::view(&ds)` checks it (reporting every problem) and borrows the fields. Use it as
/// `annoray::schema::Schema`.
///
/// Fields are variables (data variables, or coordinates with `coord`) or, with `attr`,
/// dataset attributes. A variable field is an `ArrayView` of its element type and
/// dimensionality (`ArrayView2<'a, f32>`), a `&'a VariableBase<S>` for the unit with the
/// values, or `Option` of either for a variable that may be absent; an attribute field is
/// `&'a str`, `String`, `f64`, `i64`, `&'a AttrValue`, or `Option` of one.
///
/// | Field option | Means |
/// |---|---|
/// | `coord` | a coordinate (default: a data variable) |
/// | `attr` | a dataset attribute |
/// | `name = "…"` | the variable's or attribute's name (default: the field's) |
/// | `dims = ["y", "x"]` | its dimensions, in order |
/// | `unit = "…"` | its unit (equal as units compare) |
/// | `convertible = "…"` | a unit convertible to this one |
/// | `dimension = Length` | a unit of this dimension (`annoray::dims`) |
/// | `no_unit` | no unit |
///
/// On the struct: `attr("name")`, `attr_eq("name", value)` and `attr_in("name", value, …)`
/// for dataset attributes, and `closed` to allow no variables but the fields'.
///
/// ```ignore
/// #[derive(annoray::schema::Schema)]
/// #[schema(attr_eq("format", "prism-map"))]
/// struct MapData<'a> {
///     #[schema(coord, dims = ["x"], dimension = Length)]
///     x: ArrayView1<'a, f64>,
///     #[schema(dims = ["y", "x"], unit = "1")]
///     intensity: ArrayView2<'a, f32>,
///     mask: Option<ArrayView2<'a, bool>>,
///     #[schema(attr)]
///     version: &'a str,
/// }
/// let map = MapData::view(&ds)?;
/// ```
#[proc_macro_derive(Schema, attributes(schema))]
pub fn derive_schema(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    schema::expand(&input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Select by position or by dimension name with Python-style slices, returning a view.
///
/// ```ignore
/// let v = isel!(var[0, 10:20:2, ::-1])?;              // positional
/// let v = isel!(var[-1, ..])?;                         // `..` = as many whole axes as needed
/// let v = isel!(var[time = -1, lat = 10:20])?;        // named; other dims kept whole
/// let v = isel!(mut var[time = 0:12])?;               // mutable view
/// let v = isel!(var[time = i:i + n, lon = {a::B}])?;  // expressions; wrap paths in braces
/// ```
///
/// Expands to a `Result` and never panics: out-of-range integers, zero steps, unknown
/// dimensions and overflowing bounds are all returned as errors.
#[proc_macro]
pub fn isel(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_isel(input.into())
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// One parsed indexer term.
enum Term {
    Ellipsis(Span),
    Expr(Expr),
    Slice(Span, Vec<Option<Expr>>),
}

struct Item {
    name: Option<Ident>,
    term: Term,
}

/// Parse `[mut] receiver[item, item, ...]`.
fn parse_invocation(input: TokenStream) -> syn::Result<(bool, Expr, Vec<Item>)> {
    let mut tokens = input.into_iter().peekable();
    let mutable = matches!(tokens.peek(), Some(TokenTree::Ident(i)) if i == "mut");
    if mutable {
        tokens.next();
    }
    let mut tokens: Vec<TokenTree> = tokens.collect();
    let group = match tokens.pop() {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Bracket => g,
        Some(other) => return Err(Error::new(other.span(), "expected `receiver[indexers]`")),
        None => {
            return Err(Error::new(
                Span::call_site(),
                "expected `receiver[indexers]`",
            ));
        }
    };
    if tokens.is_empty() {
        return Err(Error::new(group.span(), "missing receiver before `[...]`"));
    }
    let receiver: Expr = syn::parse2(tokens.into_iter().collect())?;
    let mut parts = split_top(group.stream(), ',');
    // `recv[]` has no indexers, and a trailing comma leaves one empty part at the end.
    if parts.last().is_some_and(Vec::is_empty) {
        parts.pop();
    }
    let items = parts
        .into_iter()
        .map(parse_item)
        .collect::<syn::Result<Vec<_>>>()?;
    Ok((mutable, receiver, items))
}

/// Named items, checked: no positional items, no repeats, no `..`.
fn named_items(items: &[Item]) -> syn::Result<Vec<(&Ident, &Term)>> {
    let mut out: Vec<(&Ident, &Term)> = Vec::new();
    for it in items {
        let Some(name) = &it.name else {
            return Err(Error::new(
                it.term.span(),
                "cannot mix named (`dim = ...`) and positional indexers",
            ));
        };
        if out.iter().any(|(n, _)| *n == name) {
            return Err(Error::new(
                name.span(),
                format!("dimension `{name}` is indexed more than once"),
            ));
        }
        if let Term::Ellipsis(span) = it.term {
            return Err(Error::new(
                span,
                "`..` is only valid in positional indexing",
            ));
        }
        out.push((name, &it.term));
    }
    Ok(out)
}

/// `match collect(items) { Ok(items) => receiver.method(items), Err(e) => Err(e) }`, using
/// method-call syntax so the receiver auto-derefs (`&T`, `&mut T`, `Box<T>`, …).
fn dispatch(
    receiver: &Expr,
    collect: TokenStream,
    tr: TokenStream,
    method: TokenStream,
) -> TokenStream {
    let rt = quote!(::annoray::__rt);
    quote! {
        match #rt::#collect(__xr_items) {
            ::core::result::Result::Ok(__xr_items) => {
                use #rt::#tr as _;
                (#receiver).#method(__xr_items)
            }
            ::core::result::Result::Err(e) => ::core::result::Result::Err(e),
        }
    }
}

fn expand_isel(input: TokenStream) -> syn::Result<TokenStream> {
    let (mutable, receiver, items) = parse_invocation(input)?;
    let rt = quote!(::annoray::__rt);
    let tr = if mutable {
        quote!(IselMut)
    } else {
        quote!(Isel)
    };
    let (pos_method, named_method) = if mutable {
        (quote!(__xr_isel_pos_mut), quote!(__xr_isel_named_mut))
    } else {
        (quote!(__xr_isel_pos), quote!(__xr_isel_named))
    };

    if items.iter().all(|it| it.name.is_none()) {
        let terms = items.iter().map(|it| term_tokens(&it.term, &rt));
        let call = dispatch(&receiver, quote!(collect_pos), tr, pos_method);
        return Ok(quote! {{
            #[allow(unused_braces)] let __xr_items = [#(#terms),*];
            #call
        }});
    }
    let pairs = named_items(&items)?.into_iter().map(|(name, term)| {
        let key = name.to_string();
        let term = term_tokens(term, &rt);
        quote!((#key, #term))
    });
    let call = dispatch(&receiver, quote!(collect_named), tr, named_method);
    Ok(quote! {{
        #[allow(unused_braces)] let __xr_items = [#(#pairs),*];
        #call
    }})
}

/// Select by coordinate label, returning a view. Dimensions are always named.
///
/// ```ignore
/// let v = sel!(ds[time = "2020-01-01":"2020-06-30", lat = 30.0:60.0])?;  // inclusive slices
/// let v = sel!(ds[time = "2020-01"])?;                                     // partial date: all of January
/// let v = sel!(da[lat = nearest(45.2), lon = nearest(10.0, tol = 0.5)])?;
/// let v = sel!(da[time = pad(t), height = {Label::quantity(2.0, "km")?}])?;
/// let v = sel!(mut ds[lat = 0.0:10.0])?;                                   // mutable view
/// ```
///
/// Scalars drop the dimension; slices (`a:b`, `a:b:step`, or Rust's `a..=b`) keep it and are
/// inclusive at both ends, as in xarray. Labels with units are converted into the
/// coordinate's unit. Expands to a `Result` and never panics.
#[proc_macro]
pub fn sel(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand_sel(input.into())
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn expand_sel(input: TokenStream) -> syn::Result<TokenStream> {
    let (mutable, receiver, items) = parse_invocation(input)?;
    let rt = quote!(::annoray::__rt);
    if let Some(it) = items.iter().find(|it| it.name.is_none()) {
        return Err(Error::new(
            it.term.span(),
            "sel! needs named dimensions: `dim = label`",
        ));
    }
    let pairs = named_items(&items)?
        .into_iter()
        .map(|(name, term)| {
            let key = name.to_string();
            let term = label_term_tokens(term, &rt)?;
            Ok(quote!((#key, #term)))
        })
        .collect::<syn::Result<Vec<_>>>()?;
    let (tr, method) = if mutable {
        (quote!(SelMut), quote!(__xr_sel_mut))
    } else {
        (quote!(Sel), quote!(__xr_sel))
    };
    let call = dispatch(&receiver, quote!(collect_labels), tr, method);
    Ok(quote! {{
        #[allow(unused_braces)] let __xr_items = [#(#pairs),*];
        #call
    }})
}

fn label_term_tokens(term: &Term, rt: &TokenStream) -> syn::Result<TokenStream> {
    Ok(match term {
        Term::Ellipsis(span) => return Err(Error::new(*span, "`..` is not a label")),
        Term::Expr(e) => match method_call(e)? {
            Some((method, label, tol)) => {
                let tol = match tol {
                    Some(t) => {
                        quote_spanned!(t.span()=> ::core::option::Option::Some(::core::convert::Into::<#rt::LabelType>::into(#t)))
                    }
                    None => quote!(::core::option::Option::None),
                };
                quote_spanned!(e.span()=> #rt::method(#rt::MethodType::#method, #label, #tol))
            }
            None => quote_spanned!(e.span()=> #rt::label(#e)),
        },
        Term::Slice(span, parts) => {
            let mut parts = parts.iter();
            let mut bound = || match parts.next() {
                Some(Some(e)) => quote_spanned!(e.span()=> #rt::some_label(#e)),
                _ => quote!(#rt::no_label()),
            };
            let (start, stop) = (bound(), bound());
            let step = match parts.next() {
                Some(Some(e)) => quote_spanned!(e.span()=> #rt::bound(#e)),
                _ => quote!(#rt::omitted()),
            };
            quote_spanned!(*span=> #rt::label_slice(#start, #stop, #step))
        }
    })
}

/// Recognise `nearest(x)`, `nearest(x, tol = t)`, `nearest(x, t)`, `pad(x)`, `backfill(x)`.
fn method_call(e: &Expr) -> syn::Result<Option<(Ident, &Expr, Option<&Expr>)>> {
    let Expr::Call(call) = e else { return Ok(None) };
    let Expr::Path(path) = &*call.func else {
        return Ok(None);
    };
    let Some(func) = path.path.get_ident() else {
        return Ok(None);
    };
    let method = match func.to_string().as_str() {
        "nearest" => "Nearest",
        "pad" | "ffill" => "Pad",
        "backfill" | "bfill" => "Backfill",
        _ => return Ok(None),
    };
    let mut args = call.args.iter();
    let label = args
        .next()
        .ok_or_else(|| Error::new(call.span(), "missing label"))?;
    let tol = match args.next() {
        None => None,
        Some(Expr::Assign(a)) => match &*a.left {
            Expr::Path(p) if p.path.is_ident("tol") || p.path.is_ident("tolerance") => {
                Some(&*a.right)
            }
            other => return Err(Error::new(other.span(), "expected `tol = ...`")),
        },
        Some(other) => Some(other),
    };
    if method != "Nearest" && tol.is_some() {
        return Err(Error::new(call.span(), "only `nearest` takes a tolerance"));
    }
    if let Some(extra) = args.next() {
        return Err(Error::new(extra.span(), "too many arguments"));
    }
    Ok(Some((Ident::new(method, func.span()), label, tol)))
}

impl Term {
    fn span(&self) -> Span {
        match self {
            Self::Ellipsis(s) | Self::Slice(s, _) => *s,
            Self::Expr(e) => e.span(),
        }
    }
}

fn term_tokens(term: &Term, rt: &TokenStream) -> TokenStream {
    match term {
        Term::Ellipsis(span) => quote_spanned!(*span=> #rt::ellipsis()),
        Term::Expr(e) => quote_spanned!(e.span()=> #rt::ix(#e)),
        Term::Slice(span, parts) => {
            let bounds = parts.iter().map(|p| match p {
                Some(e) => quote_spanned!(e.span()=> #rt::bound(#e)),
                None => quote!(#rt::omitted()),
            });
            // Pad to exactly three parts: `a:b` has no step.
            let pad = (parts.len()..3).map(|_| quote!(#rt::omitted()));
            let all = bounds.chain(pad);
            quote_spanned!(*span=> #rt::slice(#(#all),*))
        }
    }
}

fn parse_item(tokens: Vec<TokenTree>) -> syn::Result<Item> {
    let first_span = tokens.first().map_or_else(Span::call_site, TokenTree::span);
    if tokens.is_empty() {
        return Err(Error::new(first_span, "empty indexer"));
    }
    // `name = term`, where `=` is a lone `=` (not `==` or `=>`).
    let is_named = matches!(
        (tokens.first(), tokens.get(1)),
        (Some(TokenTree::Ident(_)), Some(TokenTree::Punct(p))) if p.as_char() == '=' && p.spacing() == Spacing::Alone
    );
    let (name, term_tokens) = if is_named {
        let mut it = tokens.into_iter();
        let name = match it.next() {
            Some(TokenTree::Ident(i)) => Some(i),
            _ => None,
        };
        it.next(); // the `=`
        (name, it.collect::<Vec<_>>())
    } else {
        (None, tokens)
    };
    if term_tokens.is_empty() {
        return Err(Error::new(first_span, "missing indexer after `=`"));
    }
    Ok(Item {
        name,
        term: parse_term(term_tokens)?,
    })
}

fn parse_term(tokens: Vec<TokenTree>) -> syn::Result<Term> {
    let span = tokens.first().map_or_else(Span::call_site, TokenTree::span);
    let all_dots = tokens
        .iter()
        .all(|t| matches!(t, TokenTree::Punct(p) if p.as_char() == '.'));
    if all_dots && (tokens.len() == 2 || tokens.len() == 3) {
        return Ok(Term::Ellipsis(span));
    }
    check_ambiguous_paths(&tokens)?;
    let mut parts = split_top(tokens.into_iter().collect(), ':');
    if parts.len() == 1 {
        let only = parts.pop().unwrap_or_default();
        return Ok(Term::Expr(syn::parse2(only.into_iter().collect())?));
    }
    if parts.len() > 3 {
        return Err(Error::new(
            span,
            "a slice has at most three parts: `start:stop:step`",
        ));
    }
    let parts = parts
        .into_iter()
        .map(|p| {
            if p.is_empty() {
                Ok(None)
            } else {
                syn::parse2(p.into_iter().collect()).map(Some)
            }
        })
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(Term::Slice(span, parts))
}

/// Reject `a::b`: it is either a path or the slice `a: :b`, and guessing wrong would be silent.
fn check_ambiguous_paths(tokens: &[TokenTree]) -> syn::Result<()> {
    for w in tokens.windows(4) {
        if let [prev, TokenTree::Punct(c1), TokenTree::Punct(c2), next] = w {
            let path_like = matches!(prev, TokenTree::Ident(_))
                && c1.as_char() == ':'
                && c1.spacing() == Spacing::Joint
                && c2.as_char() == ':'
                && (matches!(next, TokenTree::Ident(_))
                    || matches!(next, TokenTree::Punct(p) if p.as_char() == '<'));
            if path_like {
                return Err(Error::new(
                    c1.span(),
                    "ambiguous `::` in an indexer: write `{a::b}` for a path, or `a: :b` for a slice with no stop",
                ));
            }
        }
    }
    Ok(())
}

/// Split a token stream on a punctuation character at the top level. Delimited groups are
/// single token trees, so separators inside `(...)`, `[...]` and `{...}` are left alone.
fn split_top(stream: TokenStream, sep: char) -> Vec<Vec<TokenTree>> {
    let mut parts = vec![Vec::new()];
    for tt in stream {
        if matches!(&tt, TokenTree::Punct(p) if p.as_char() == sep) {
            parts.push(Vec::new());
        } else if let Some(last) = parts.last_mut() {
            last.push(tt);
        }
    }
    parts
}
