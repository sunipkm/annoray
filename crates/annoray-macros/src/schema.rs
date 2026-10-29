//! `#[derive(Schema)]`: a struct of typed views of a dataset, and its schema.

use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Error, ExprArray, Fields, Lit, LitStr, Path, Token};

/// What a field is: a data variable, a coordinate or a dataset attribute.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Data,
    Coord,
    Attr,
}

/// A field and what its `#[schema(...)]` says.
struct Field {
    ident: syn::Ident,
    ty: syn::Type,
    role: Role,
    name: String,
    /// `Var` builder calls (`.dims([...])`, `.unit("1")`, …).
    var: Vec<TokenStream>,
}

/// A struct-level rule: a `Schema` builder call.
fn struct_rules(input: &DeriveInput) -> syn::Result<Vec<TokenStream>> {
    let mut rules = Vec::new();
    for a in input.attrs.iter().filter(|a| a.path().is_ident("schema")) {
        a.parse_nested_meta(|meta| {
            let key = meta
                .path
                .get_ident()
                .map(ToString::to_string)
                .unwrap_or_default();
            match key.as_str() {
                "closed" => rules.push(quote!(.closed())),
                "attr" | "attr_eq" | "attr_in" => {
                    let content;
                    syn::parenthesized!(content in meta.input);
                    let name: LitStr = content.parse()?;
                    let mut values: Vec<Lit> = Vec::new();
                    while content.parse::<Option<Token![,]>>()?.is_some() && !content.is_empty() {
                        values.push(content.parse()?);
                    }
                    rules.push(match (key.as_str(), values.as_slice()) {
                        ("attr", []) => quote!(.attr(#name)),
                        ("attr_eq", [v]) => quote!(.attr_eq(#name, #v)),
                        ("attr_in", [_, ..]) => quote!(.attr_in(#name, [#(#values),*])),
                        _ => {
                            return Err(meta.error(
                                "expected attr(\"name\"), attr_eq(\"name\", value) or \
                                 attr_in(\"name\", value, …)",
                            ));
                        }
                    });
                }
                _ => {
                    return Err(meta.error(
                        "unknown schema option (expected closed, attr, attr_eq or attr_in)",
                    ));
                }
            }
            Ok(())
        })?;
    }
    Ok(rules)
}

fn field(f: &syn::Field) -> syn::Result<Field> {
    let ident = f
        .ident
        .clone()
        .ok_or_else(|| Error::new(f.span(), "Schema needs named fields"))?;
    let mut out = Field {
        name: ident.to_string().trim_start_matches("r#").to_owned(),
        ident,
        ty: f.ty.clone(),
        role: Role::Data,
        var: Vec::new(),
    };
    for a in f.attrs.iter().filter(|a| a.path().is_ident("schema")) {
        a.parse_nested_meta(|meta| {
            let key = meta
                .path
                .get_ident()
                .map(ToString::to_string)
                .unwrap_or_default();
            match key.as_str() {
                "coord" => out.role = Role::Coord,
                "attr" => out.role = Role::Attr,
                "name" => out.name = meta.value()?.parse::<LitStr>()?.value(),
                "dims" => {
                    let dims: ExprArray = meta.value()?.parse()?;
                    let dims = dims.elems.iter();
                    out.var
                        .push(quote!(.and(::annoray::schema::Var::new([#(#dims),*]))));
                }
                "unit" | "convertible" => {
                    let u: LitStr = meta.value()?.parse()?;
                    let m = syn::Ident::new(&key, meta.path.span());
                    out.var.push(quote!(.#m(#u)));
                }
                "dimension" => {
                    let d: Path = meta.value()?.parse()?;
                    out.var.push(quote!(.dimension::<#d>()));
                }
                "no_unit" => out.var.push(quote!(.no_unit())),
                _ => {
                    return Err(meta.error(
                        "unknown schema option (expected coord, attr, name, dims, unit, \
                         convertible, dimension or no_unit)",
                    ));
                }
            }
            Ok(())
        })?;
    }
    if out.role == Role::Attr
        && let Some(v) = out.var.first()
    {
        return Err(Error::new(v.span(), "an attribute field takes only `name`"));
    }
    Ok(out)
}

pub(crate) fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let Data::Struct(data) = &input.data else {
        return Err(Error::new(input.span(), "Schema is derived for structs"));
    };
    let Fields::Named(named) = &data.fields else {
        return Err(Error::new(input.span(), "Schema needs named fields"));
    };
    let lifetime = input
        .generics
        .lifetimes()
        .next()
        .map(|l| l.lifetime.clone());
    let Some(a) = lifetime else {
        return Err(Error::new(
            input.generics.span(),
            "Schema needs a lifetime parameter (`struct Name<'a>`): its fields borrow from the \
             dataset",
        ));
    };
    let fields = named
        .named
        .iter()
        .map(field)
        .collect::<syn::Result<Vec<_>>>()?;
    let rules = struct_rules(input)?;
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let mut builds = Vec::new();
    let mut bounds = Vec::new();
    let mut inits = Vec::new();
    for f in &fields {
        let (ty, name, id) = (&f.ty, &f.name, &f.ident);
        let span = ty.span();
        match f.role {
            Role::Attr => {
                builds.push(quote_spanned! {span=>
                    let s = <#ty as ::annoray::schema::AttrField<#a>>::require(s, #name);
                });
                inits.push(quote_spanned! {span=>
                    #id: <#ty as ::annoray::schema::AttrField<#a>>::extract(ds.attrs().get(#name), #name)?
                });
            }
            Role::Data | Role::Coord => {
                let var = &f.var;
                let (required, optional, repr, lookup) = if f.role == Role::Coord {
                    (
                        quote!(coord),
                        quote!(optional_coord),
                        quote!(<S as ::annoray::Repr>::Coord),
                        quote!(ds.coords()),
                    )
                } else {
                    (
                        quote!(var),
                        quote!(optional),
                        quote!(S),
                        quote!(ds.data_vars()),
                    )
                };
                builds.push(quote_spanned! {span=>
                    let v = <#ty as ::annoray::schema::FieldSpec>::spec() #(#var)*;
                    let s = if <#ty as ::annoray::schema::FieldSpec>::OPTIONAL {
                        s.#optional(#name, v)
                    } else {
                        s.#required(#name, v)
                    };
                });
                bounds.push(quote_spanned! {span=> #ty: ::annoray::schema::VarField<#a, #repr> });
                inits.push(quote_spanned! {span=>
                    #id: <#ty as ::annoray::schema::VarField<#a, #repr>>::extract(#lookup.get(#name), #name)?
                });
            }
        }
    }
    let doc_view = format!(
        "Checks `ds` against [`{ident}::schema`], then borrows the fields from it: views of the \
         values (lazy variables are read now, and kept) and attributes.\n\n# Errors\n\
         [`annoray::Error::Schema`] with every problem found; read errors of lazy values."
    );
    Ok(quote! {
        impl #impl_generics #ident #ty_generics #where_clause {
            /// What a dataset must hold to be viewed as this struct.
            #[must_use]
            pub fn schema() -> ::annoray::schema::Schema {
                let s = ::annoray::schema::Schema::new() #(#rules)*;
                #(#builds)*
                s
            }

            #[doc = #doc_view]
            pub fn view<S: ::annoray::Repr>(
                ds: &#a ::annoray::DatasetBase<S>,
            ) -> ::core::result::Result<Self, ::annoray::Error>
            where
                #(#bounds,)*
            {
                Self::schema().check(ds)?;
                ::core::result::Result::Ok(Self { #(#inits,)* })
            }
        }
    })
}
