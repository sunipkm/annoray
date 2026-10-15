"""Quantities: numbers with units, checked when combined.

The Rust twin (`annoray/examples/quantities.rs`) checks dimensions at compile time; this
one at run time. Both print the same lines.

    python examples/quantities.py
"""

import annoray as ar
from annoray import Quantity as Q


def yes(b):
    return "yes" if b else "no"


# A length over a time is a velocity.
d = Q(2.5, "km")
t = Q(30.0, "min")
v = d / t
print(f"speed: {v:.3f}")
print(f"speed: {v.to('m/s'):.3f}")

# Composite units come out of arithmetic. (Dividing by s once more would give m s-4: Python
# computes it; the Rust twin does not compile, as the result would not be a jerk.)
s = t.to("s")
j = v.to("m/s") / s / s
print(f"jerk unit: {j.units}")
print(f"jerk: {j.to('um s-3'):.3f}")

# Physics reads like physics.
m = Q(1200.0, "kg")
car = Q(27.8, "m/s")
e = m * car * car * 0.5
print(f"kinetic energy: {e.to('kJ'):.3f}")
p = Q(500.0, "N") / Q(25.0, "cm2")
print(f"pressure: {p.to('kPa'):.3f}")

# Units are values: same dimension, different scales need an explicit conversion.
km = Q(1.0, "km")
one_m = Q(1.0, "m")
try:
    km + one_m
    print("1 km + 1 m -> (no error?)")
except ar.UnitError as err:
    print(f"1 km + 1 m -> error: {err}")
print(f"1 km + 1 m -> {km + one_m.to('km'):.3f}")
print(f"1 km == 1000 m: {yes(km == Q(1000.0, 'm'))}")

# Absolute temperatures: differences and conversions work; products do not.
warm = Q(20.0, "degC")
cool = Q(10.0, "degC")
print(f"20 degC - 10 degC = {warm - cool:.1f}")
print(f"20 degC = {warm.to('degF'):.1f}")
try:
    warm * 2.0
    failed = False
except ar.UnitError:
    failed = True
print(f"2 x 20 degC -> error: {yes(failed)}")

# A unit known only at run time (from a file, say).
read = Q(12.0, "m s-1")
print(f"is a velocity: {yes(read.unit.is_convertible('m/s'))}")
print(f"is a length: {yes(read.unit.is_convertible('m'))}")
