//! JSON with Python's non-finite floats. zarr-python writes attributes with Python's `json`,
//! which spells NaN and the infinities as the bare tokens `NaN`, `Infinity` and `-Infinity`:
//! not JSON, so a strict parser refuses the whole file. Reading turns each bare token into a
//! marker string (a NUL, then the token: no text a writer means holds that), which
//! [`special_float`] reads back; writing turns markers back into bare tokens, so attributes
//! round-trip with zarr-python and a string `"NaN"` stays a string.

use std::borrow::Cow;

use serde_json::Value;

/// The bare tokens, with their values.
const TOKENS: [(&str, f64); 3] = [
    ("-Infinity", f64::NEG_INFINITY),
    ("Infinity", f64::INFINITY),
    ("NaN", f64::NAN),
];

fn marker(token: &str) -> String {
    format!("\u{0}{token}")
}

/// The non-finite float a marker stands for (`None` for any other value).
#[must_use]
pub fn special_float(v: &Value) -> Option<f64> {
    let s = v.as_str()?.strip_prefix('\u{0}')?;
    TOKENS.iter().find(|(t, _)| *t == s).map(|(_, f)| *f)
}

/// A float as an attribute value: a number, or for NaN and the infinities a marker, written
/// as the bare token.
#[must_use]
pub fn float_value(f: f64) -> Value {
    if f.is_finite() {
        serde_json::Number::from_f64(f).map_or(Value::Null, Value::Number)
    } else {
        let token = if f.is_nan() {
            "NaN"
        } else if f > 0.0 {
            "Infinity"
        } else {
            "-Infinity"
        };
        Value::String(marker(token))
    }
}

/// `text` with every bare token outside string literals quoted as its marker.
pub(crate) fn quote_bare_tokens(text: &[u8]) -> Cow<'_, [u8]> {
    // Most files have none: no copy then.
    let candidate = |b: u8| b == b'N' || b == b'I';
    if !text.iter().copied().any(candidate) {
        return Cow::Borrowed(text);
    }
    let mut out = Vec::with_capacity(text.len());
    let (mut in_string, mut escaped) = (false, false);
    let mut i = 0;
    while let Some(&b) = text.get(i) {
        if in_string {
            out.push(b);
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            i = i.saturating_add(1);
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push(b);
            i = i.saturating_add(1);
            continue;
        }
        let rest = text.get(i..).unwrap_or_default();
        if let Some((token, _)) = TOKENS.iter().find(|(t, _)| rest.starts_with(t.as_bytes())) {
            // The marker, as a JSON string (the NUL escaped).
            out.extend_from_slice(format!("\"\\u0000{token}\"").as_bytes());
            i = i.saturating_add(token.len());
            continue;
        }
        out.push(b);
        i = i.saturating_add(1);
    }
    Cow::Owned(out)
}

/// Serialized JSON with every marker string written as its bare token.
pub(crate) fn unquote_markers(mut text: String) -> String {
    for (token, _) in TOKENS {
        text = text.replace(&format!("\"\\u0000{token}\""), token);
    }
    text
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn bare_tokens_round_trip() {
        let text =
            br#"{"a": NaN, "b": [1.0, -Infinity, Infinity], "s": "NaN", "t": "x NaN \" NaN"}"#;
        let quoted = quote_bare_tokens(text);
        let v: Value = serde_json::from_slice(&quoted).unwrap();
        assert!(special_float(&v["a"]).unwrap().is_nan());
        assert_eq!(special_float(&v["b"][1]), Some(f64::NEG_INFINITY));
        assert_eq!(special_float(&v["b"][2]), Some(f64::INFINITY));
        assert_eq!(v["s"], Value::String("NaN".into()));
        assert_eq!(v["t"], Value::String("x NaN \" NaN".into()));
        assert_eq!(special_float(&v["s"]), None);
        let back = unquote_markers(serde_json::to_string(&v).unwrap());
        assert_eq!(
            back,
            r#"{"a":NaN,"b":[1.0,-Infinity,Infinity],"s":"NaN","t":"x NaN \" NaN"}"#
        );
        assert!(matches!(quote_bare_tokens(b"{\"a\": 1}"), Cow::Borrowed(_)));
        assert_eq!(float_value(2.5), serde_json::json!(2.5));
    }
}
