"""Extract the distinct canonical units from the CF standard-name table.

Run from the repository root:
    python3 fixtures/gen_cf_units.py cf-standard-name-table.xml > fixtures/cf_canonical_units.txt

The table is https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml
"""

import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
print(f"# CF standard name table version {root.findtext('version_number')}")
for u in sorted({(e.findtext("canonical_units") or "").strip() for e in root.iter("entry")}):
    print(u)
