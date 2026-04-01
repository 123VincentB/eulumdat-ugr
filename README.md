# eulumdat-ugr

[![PyPI version](https://img.shields.io/pypi/v/eulumdat-ugr.svg)](https://pypi.org/project/eulumdat-ugr/)
[![Python](https://img.shields.io/pypi/pyversions/eulumdat-ugr.svg)](https://pypi.org/project/eulumdat-ugr/)
[![License: MIT](https://img.shields.io/github/license/123VincentB/eulumdat-ugr)](https://github.com/123VincentB/eulumdat-ugr/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/1184236539.svg)](https://doi.org/10.5281/zenodo.19367173)

UGR (Unified Glare Rating) calculation from EULUMDAT (.ldt) photometric files — part of the [eulumdat-*](https://pypi.org/project/eulumdat-py/) ecosystem.

Produces the full **19 × 10 UGR table** per CIE 117:1995 and CIE 190:2010, compatible with DIALux and Relux catalogue output.

## Installation

```bash
pip install eulumdat-ugr
```

## Quick start

```python
from pyldt import LdtReader
from eulumdat_ugr import UgrCalculator

ldt = LdtReader.read("luminaire.ldt")
result = UgrCalculator.compute(ldt)

# NumPy array (19, 10)
print(result.values)

# Export to CSV
with open("ugr_table.csv", "w") as f:
    f.write(result.to_csv())

# Export to JSON (compact)
with open("ugr_table.json", "w") as f:
    f.write(result.to_json())

# Export to JSON (human-readable)
with open("ugr_table.json", "w") as f:
    f.write(result.to_json(indent=2))
```

## Output format

`result.values` is a **(19 × 10)** NumPy array:

- **19 rows** — standard room configurations from CIE 190:2010 (2H×2H to 12H×8H)
- **Columns 0–4** — crosswise orientation, 5 reflectance combinations
- **Columns 5–9** — endwise orientation, same 5 reflectance combinations

Reflectance combinations (col % 5):

| Index | Ceiling | Walls | Reference plane |
|-------|---------|-------|-----------------|
| 0     | 0.7     | 0.5   | 0.2             |
| 1     | 0.7     | 0.3   | 0.2             |
| 2     | 0.5     | 0.5   | 0.2             |
| 3     | 0.5     | 0.3   | 0.2             |
| 4     | 0.3     | 0.3   | 0.2             |

## Standards

- **CIE 117:1995** — UGR formula, Guth position index (Table 4.1)
- **CIE 190:2010** — tabular method, background luminance (E_WID), standard room geometry

## Validation

Validated against DIALux and Relux reference tables on 11 luminaire samples (SHR = 0.25):

| Reference | Max deviation |
|-----------|---------------|
| Relux     | ≤ 0.43 UGR    |
| DIALux    | ≤ 1.01 UGR    |

DIALux values are systematically 0.1–0.8 UGR below Relux for small rooms (k ≤ 2.0) — a known inter-software variation. Our implementation follows CIE 190:2010 and aligns with Relux.

## Dependencies

- [eulumdat-py](https://pypi.org/project/eulumdat-py/) >= 1.0.0
- [eulumdat-luminance](https://pypi.org/project/eulumdat-luminance/) >= 1.2.0
- numpy >= 1.24
- scipy >= 1.10

## Geometry

Fixed geometry per CIE 190:2010 §4.2:

- H = 2.0 m (luminaire height above observer eye level)
- Observer at mid-point of short wall, eye level 1.2 m
- SHR = 0.25 (standard catalogue spacing)
