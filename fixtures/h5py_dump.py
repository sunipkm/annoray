"""Print h5py's canonical dump of an HDF5 file (same format as gen_hdf5.py's .dump files).

Usage: python3 fixtures/h5py_dump.py FILE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_hdf5

path = sys.argv[1]
tmp = path + ".h5"
os.replace(path, tmp) if not path.endswith(".h5") else None
target = tmp if not path.endswith(".h5") else path
gen_hdf5.dump(target)
with open(target.replace(".h5", ".dump")) as f:
    sys.stdout.write(f.read())
