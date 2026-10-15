"""Quantities: numbers with units, their dimensions explicit and checked.

The Rust twin (`annoray/examples/quantities.rs`) checks dimensions when it compiles; this
one when it runs. Both print the same lines.

    python examples/quantities.py
"""

import annoray as ar
from annoray import Quantity
from annoray.dims import Area, Dim, Force, Length, Mass, Temperature, Time, Velocity, checked


class Jerk(Dim, dimension=Length / Time**3):
    """Rate of change of acceleration, composed from SI dimensions.

    (Rust: `dimension! { pub Jerk = Length / Time^3; }`.)"""


def yes(b):
    return "yes" if b else "no"


@checked
def speed(d: Quantity[Length], t: Quantity[Time]) -> Quantity[Velocity]:
    return d / t


# A length over a time is a velocity: checked going in and coming out.
d = Length(2.5, "km")
t = Time(30.0, "min")
v = speed(d, t)
print(f"speed: {v:.3f}")
print(f"speed: {v.to('m/s'):.3f}")

# A velocity over a time twice is a jerk. (Dividing once more would be m s-4: the Rust twin
# does not compile; here `Jerk.check` raises.)
s = t.to("s")
j = Jerk.check(v.to("m/s") / s / s)
print(f"jerk unit: {j.units}")
print(f"jerk: {j.to('um s-3'):.3f}")

# Physics reads like physics.
m = Mass(1200.0, "kg")
car = Velocity(27.8, "m/s")
e = m * car * car * 0.5
print(f"kinetic energy: {e.to('kJ'):.3f}")
p = Force(500.0, "N") / Area(25.0, "cm2")
print(f"pressure: {p.to('kPa'):.3f}")

# Units are values: same dimension, different scales need an explicit conversion.
km = Length(1.0, "km")
one_m = Length(1.0, "m")
try:
    _ = km + one_m
    print("1 km + 1 m -> (no error?)")
except ar.UnitError as err:
    print(f"1 km + 1 m -> error: {err}")
print(f"1 km + 1 m -> {km + one_m.to('km'):.3f}")
print(f"1 km == 1000 m: {yes(km == Length(1000.0, 'm'))}")

# Absolute temperatures: differences and conversions work; products do not.
warm = Temperature(20.0, "degC")
cool = Temperature(10.0, "degC")
print(f"20 degC - 10 degC = {warm - cool:.1f}")
print(f"20 degC = {warm.to('degF'):.1f}")
try:
    _ = warm * 2.0
    failed = False
except ar.UnitError:
    failed = True
print(f"2 x 20 degC -> error: {yes(failed)}")

# A unit known only at run time (from a file, say).
read = ar.Quantity(12.0, "m s-1")
print(f"is a velocity: {yes(isinstance(read, Velocity))}")
print(f"is a length: {yes(isinstance(read, Length))}")
