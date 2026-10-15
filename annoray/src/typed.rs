//! Arrays of compile-time quantities (`uom`, `whippyunits`; see [`StaticUnit`]):
//! [`DataArrayBase::to_typed`], [`DataArrayBase::view_typed`], [`DataArray::from_typed`], and
//! typed labels for `sel`.

use annoray_units::{StaticUnit, StaticValue, Unit, converter_from, converter_into};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

use crate::{CoreError, DataArray, DataArrayBase, Element, Name, Repr, Result};

fn unit_or_one(u: Option<&Unit>) -> Unit {
    u.cloned().unwrap_or_else(Unit::one)
}

impl<S: Repr> DataArrayBase<S> {
    /// The values as compile-time quantities `Q`, converted into `Q`'s unit (a copy). An
    /// array without a unit counts as dimensionless.
    ///
    /// ```
    /// # #[cfg(feature = "whippyunits")] {
    /// use annoray::DataArray;
    /// use annoray::ndarray::array;
    /// use whippyunits::qty;
    ///
    /// let h = DataArray::from_array(["x"], array![1500.0, 250.0].into_dyn())?
    ///     .with_unit("m".parse()?);
    /// let km = h.to_typed::<qty!(km)>()?;
    /// assert_eq!(km[[0]].unsafe_value, 1.5);
    /// # }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// [`CoreError::Unit`] if the unit measures something else than `Q`; errors of
    /// [`to_f64`](Self::to_f64) for values that are not numbers.
    pub fn to_typed<Q: StaticUnit>(&self) -> Result<ArrayD<Q>> {
        let c = converter_into::<Q>(&unit_or_one(self.unit()))?;
        Ok(self
            .to_f64()?
            .mapv(|x| Q::from_raw(Q::Value::from_f64(c.apply(x)))))
    }

    /// The values viewed in place as compile-time quantities `Q`: only when the array holds
    /// `Q::Value` contiguously, in exactly `Q`'s unit, and `Q` is layout-compatible with its
    /// value (whippyunits). Lazy data is loaded first.
    ///
    /// # Errors
    /// [`CoreError::NotViewable`] unless the unit is exactly `Q`'s, the data contiguous and
    /// `Q` viewable (use [`to_typed`](Self::to_typed)); [`CoreError::DTypeMismatch`] for
    /// another element type.
    pub fn view_typed<Q>(&self) -> Result<ArrayViewD<'_, Q>>
    where
        Q: StaticUnit,
        Q::Value: Element,
    {
        let own = unit_or_one(self.unit());
        let target = Q::unit()?;
        if Q::ANGLE_DIMENSIONLESS || Q::is_difference() || !own.converter_to(&target)?.is_identity()
        {
            return Err(CoreError::NotViewable(format!(
                "a view needs the array in exactly {target} (it is in {own}); convert with to_typed"
            )));
        }
        let data = self.data::<Q::Value>()?;
        let shape = data.shape().to_vec();
        let slice = data.to_slice().ok_or_else(|| {
            CoreError::NotViewable("a view needs contiguous, row-major data".into())
        })?;
        let typed = Q::view_slice(slice).ok_or_else(|| {
            CoreError::NotViewable("this quantity type cannot be viewed in place".into())
        })?;
        ArrayViewD::from_shape(IxDyn(&shape), typed)
            .map_err(|e| CoreError::NotViewable(e.to_string()))
    }
}

impl DataArray {
    /// A data array of compile-time quantities: the raw values, in `Q`'s unit.
    ///
    /// # Errors
    /// Errors of [`StaticUnit::unit`] (a uom kind with no runtime unit); as for
    /// [`from_array`](Self::from_array).
    pub fn from_typed<Q, D>(dims: D, data: ArrayD<Q>) -> Result<Self>
    where
        Q: StaticUnit + Clone,
        Q::Value: Element,
        D: IntoIterator,
        D::Item: Into<Name>,
    {
        let unit = Q::unit()?;
        Ok(Self::from_array(dims, data.mapv(StaticUnit::into_raw))?.with_unit(unit))
    }

    /// The values of `Q` in `unit`: a typed array converted to a runtime one in a chosen unit.
    ///
    /// # Errors
    /// [`CoreError::Unit`] if `unit` measures something else; as for
    /// [`from_typed`](Self::from_typed).
    pub fn from_typed_in<Q, D>(dims: D, data: &ArrayD<Q>, unit: Unit) -> Result<Self>
    where
        Q: StaticUnit + Clone,
        D: IntoIterator,
        D::Item: Into<Name>,
    {
        let c = converter_from::<Q>(&unit)?;
        let values = data.mapv(|q| c.apply(q.into_raw().to_f64()));
        Ok(Self::from_array(dims, values)?.with_unit(unit))
    }
}

/// `sel` by compile-time quantities: `sel!(da[height = {Length::new::<kilometer>(2.0)}])`.
macro_rules! typed_labels {
    ($($(#[$attr:meta])* [$($gen:tt)*] $ty:ty;)*) => {$(
        $(#[$attr])*
        impl<$($gen)*> crate::IntoLabelIndexer for $ty
        where
            $ty: StaticUnit,
        {
            fn into_label_indexer(self) -> Result<crate::LabelIndexer> {
                let q = annoray_units::Quantity::from_static(self)?;
                crate::Label::from(q).into_label_indexer()
            }
        }
    )*};
}

typed_labels! {
    #[cfg(feature = "uom")]
    [D: uom::si::Dimension + ?Sized] uom::si::Quantity<D, uom::si::SI<f64>, f64>;
    #[cfg(feature = "uom")]
    [D: uom::si::Dimension + ?Sized] uom::si::Quantity<D, uom::si::SI<f32>, f32>;
    #[cfg(feature = "whippyunits")]
    [U, V] whippyunits::Quantity<U, V, ()>;
}
