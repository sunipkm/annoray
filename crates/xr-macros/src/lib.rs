//! Procedural macros for xarray-rs. Use them through the `xarray-rs` facade crate.

use proc_macro2::{Delimiter, Spacing, Span, TokenStream, TokenTree};
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Error, Expr, Ident};

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

fn expand_isel(input: TokenStream) -> syn::Result<TokenStream> {
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

    let named = items.iter().filter(|it| it.name.is_some()).count();
    let rt = quote!(::xarray_rs::__rt);
    let target = if mutable {
        quote!(&mut (#receiver))
    } else {
        quote!(&(#receiver))
    };

    if named == 0 {
        let terms = items.iter().map(|it| term_tokens(&it.term, &rt));
        return Ok(quote! {{
            #[allow(unused_braces)] let __xr_items = [#(#terms),*];
            #rt::isel_pos(#target, __xr_items)
        }});
    }
    if named != items.len() {
        let span = items
            .iter()
            .find(|it| it.name.is_none())
            .map_or_else(Span::call_site, |it| it.term.span());
        return Err(Error::new(
            span,
            "cannot mix named (`dim = ...`) and positional indexers",
        ));
    }

    let mut seen: Vec<&Ident> = Vec::new();
    let mut pairs = Vec::new();
    for it in &items {
        let Some(name) = &it.name else { continue };
        if seen.contains(&name) {
            return Err(Error::new(
                name.span(),
                format!("dimension `{name}` is indexed more than once"),
            ));
        }
        seen.push(name);
        if let Term::Ellipsis(span) = it.term {
            return Err(Error::new(
                span,
                "`..` is only valid in positional indexing",
            ));
        }
        let key = name.to_string();
        let term = term_tokens(&it.term, &rt);
        pairs.push(quote!((#key, #term)));
    }
    Ok(quote! {{
        #[allow(unused_braces)] let __xr_items = [#(#pairs),*];
        #rt::isel_named(#target, __xr_items)
    }})
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
