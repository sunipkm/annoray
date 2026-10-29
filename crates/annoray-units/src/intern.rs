//! [`UnitId`]: a `Copy` handle to a [`Unit`] kept, once per spelling, in a process-wide table.
//!
//! The table only grows: an entry lives as long as the process, so handles never dangle and
//! reading a unit through one takes no lock. Its size is capped ([`set_intern_cap`]), so input
//! that names many distinct units (files with ever new epochs, say) cannot exhaust memory:
//! interning past the cap is [`UnitError::InternFull`], and plain [`Unit`] values keep working.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use crate::{Unit, UnitError};

/// The default [`intern_cap`]: at most about 20 MiB of table, plus the text of the labels.
pub const DEFAULT_INTERN_CAP: usize = 65_536;

static CAP: AtomicUsize = AtomicUsize::new(DEFAULT_INTERN_CAP);

/// `1`, outside the table: always available, never counted.
static ONE: Unit = Unit::one();

#[derive(Default)]
struct Table {
    /// Every spelling of each unit (equal units, different labels).
    units: HashMap<Unit, Vec<&'static Unit>>,
    /// Parsed strings (trimmed), so a string seen before is not parsed again.
    spellings: HashMap<Box<str>, &'static Unit>,
    /// Entries made (the spellings of all units).
    len: usize,
}

fn table() -> MutexGuard<'static, Table> {
    static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();
    // Every update leaves the table whole, so a panic elsewhere while it was held is harmless.
    TABLE
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// The most units the table holds (all spellings counted).
#[must_use]
pub fn intern_cap() -> usize {
    CAP.load(Ordering::Relaxed)
}

/// Sets [`intern_cap`]. Entries already made stay; below their number, nothing new is
/// interned. Applications that read untrusted files may lower it; the default is
/// [`DEFAULT_INTERN_CAP`].
pub fn set_intern_cap(cap: usize) {
    CAP.store(cap, Ordering::Relaxed);
}

/// The number of units interned so far.
#[must_use]
pub fn interned() -> usize {
    table().len
}

/// A [`Unit`] interned once per spelling: `Copy`, the size of a pointer, and read without a
/// lock (it dereferences to the unit).
///
/// Handles to the same spelling are the same handle ([`same_spelling`](Self::same_spelling)
/// compares them as pointers). Equality and hashing are the unit's, so they ignore the label
/// as [`Unit`]'s do: the handles of `km` and `1000 m` differ but compare equal.
///
/// ```
/// use annoray_units::{Unit, UnitId};
///
/// let km = UnitId::parse("km")?;
/// assert!(km.same_spelling(UnitId::parse(" km ")?)); // found again, not parsed again
/// assert_eq!(km, UnitId::parse("1000 m")?);          // equal units, other spelling
/// assert_eq!(km.to_string(), "km");
/// assert_eq!(*km, "km".parse::<Unit>()?);
/// # Ok::<(), annoray_units::UnitError>(())
/// ```
#[derive(Clone, Copy)]
pub struct UnitId(&'static Unit);

impl UnitId {
    /// The dimensionless unit `1` (not counted against the cap).
    #[must_use]
    pub fn one() -> Self {
        Self(&ONE)
    }

    /// The handle of `unit`, as spelled (its label).
    ///
    /// # Errors
    /// [`UnitError::InternFull`] if `unit` is new and the table is at [`intern_cap`].
    pub fn intern(unit: Unit) -> Result<Self, UnitError> {
        if unit.label().is_none() && unit == ONE {
            return Ok(Self::one());
        }
        let mut t = table();
        Ok(Self(insert(&mut t, unit)?))
    }

    /// The handle of a parsed unit string ([`Unit::parse`]); a string seen before is looked up,
    /// not parsed again.
    ///
    /// # Errors
    /// As for [`Unit::parse`], and [`UnitError::InternFull`].
    pub fn parse(s: &str) -> Result<Self, UnitError> {
        let key = s.trim();
        if let Some(u) = local_get(key) {
            return Ok(Self(u));
        }
        let u = shared_parse(key)?;
        local_put(key, u);
        Ok(Self(u))
    }

    /// The unit, for as long as the process runs.
    #[must_use]
    pub fn unit(self) -> &'static Unit {
        self.0
    }

    /// Whether both handles name the same unit spelled the same way (one pointer comparison).
    #[must_use]
    pub fn same_spelling(self, other: Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

/// The most spellings a thread's cache holds before it starts again (empty).
const LOCAL_CAP: usize = 64;

thread_local! {
    /// Spellings this thread parsed recently, mapped to the shared table's entries: found
    /// without the lock. Entries are only ever made in the shared table, so every thread's
    /// cache holds the same handle for a spelling.
    static LOCAL: RefCell<HashMap<Box<str>, &'static Unit>> = RefCell::new(HashMap::new());
}

fn local_get(key: &str) -> Option<&'static Unit> {
    // (Unavailable while the thread exits: the shared table answers instead.)
    LOCAL
        .try_with(|c| c.try_borrow().ok()?.get(key).copied())
        .ok()
        .flatten()
}

fn local_put(key: &str, unit: &'static Unit) {
    let _ = LOCAL.try_with(|c| {
        if let Ok(mut c) = c.try_borrow_mut() {
            if c.len() >= LOCAL_CAP {
                c.clear();
            }
            c.insert(key.into(), unit);
        }
    });
}

/// The shared table's entry for a trimmed unit string, parsed and made if new.
fn shared_parse(key: &str) -> Result<&'static Unit, UnitError> {
    if let Some(u) = table().spellings.get(key) {
        return Ok(u);
    }
    // Parsed without the lock. Another thread may intern the same spelling meanwhile: `insert`
    // then finds its entry, and both get the same handle.
    let unit: Unit = key.parse()?;
    if unit.label().is_none() && unit == ONE {
        return Ok(&ONE);
    }
    let mut t = table();
    let u = insert(&mut t, unit)?;
    t.spellings.insert(key.into(), u);
    Ok(u)
}

/// The table's copy of `unit` (with its label), made if new.
fn insert(t: &mut Table, unit: Unit) -> Result<&'static Unit, UnitError> {
    if let Some(found) = t
        .units
        .get(&unit)
        .and_then(|v| v.iter().find(|u| u.label() == unit.label()))
    {
        return Ok(found);
    }
    let cap = intern_cap();
    if t.len >= cap {
        return Err(UnitError::InternFull { cap });
    }
    let leaked: &'static Unit = Box::leak(Box::new(unit));
    t.units
        .entry(leaked.clone().without_label())
        .or_default()
        .push(leaked);
    t.len = t.len.saturating_add(1);
    Ok(leaked)
}

impl Deref for UnitId {
    type Target = Unit;
    fn deref(&self) -> &Unit {
        self.0
    }
}

impl AsRef<Unit> for UnitId {
    fn as_ref(&self) -> &Unit {
        self.0
    }
}

impl PartialEq for UnitId {
    fn eq(&self, other: &Self) -> bool {
        self.same_spelling(*other) || self.0 == other.0
    }
}

impl Eq for UnitId {}

impl Hash for UnitId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialEq<Unit> for UnitId {
    fn eq(&self, other: &Unit) -> bool {
        self.0 == other
    }
}

impl fmt::Display for UnitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.0, f)
    }
}

impl fmt::Debug for UnitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UnitId({:?})", self.0.to_string())
    }
}

impl TryFrom<Unit> for UnitId {
    type Error = UnitError;
    fn try_from(unit: Unit) -> Result<Self, UnitError> {
        Self::intern(unit)
    }
}

impl From<UnitId> for Unit {
    fn from(id: UnitId) -> Self {
        id.0.clone()
    }
}
