# annoray-macros

Procedural macros of [annoray](https://docs.rs/annoray): `isel!`/`sel!` with Python slice
syntax. Intended to be consumed through the re-export `annoray::{isel, sel}`.

```rust,ignore
let v = isel!(var[0, 10:20:2, ::-1])?;                     // by position
let v = isel!(var[time = -1, lat = 10:20])?;               // by dimension name
let v = sel!(da[lat = 30.0:60.0, lon = nearest(260.0)])?;  // by label (inclusive slices)
isel!(mut var[time = 0:12])?.data_mut::<f64>()?.fill(0.0); // mutable views write through
```

Both expand to an expression of type `Result`; out-of-range indices, zero steps, unknown
dimensions and overflowing bounds are runtime errors, never panics.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at the licensee's
option.
