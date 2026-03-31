# Basic usage

## Compute the full UGR table

```python
from pyldt import LdtReader
from eulumdat_ugr import UgrCalculator

ldt = LdtReader.read("luminaire.ldt")
result = UgrCalculator.compute(ldt)
```

## Access the values

`result.values` is a NumPy array of shape **(19 × 10)**:

- **19 rows** — standard room configurations (CIE 190:2010)
- **10 columns** — cols 0–4 crosswise, cols 5–9 endwise
- Column index `% 5` selects the reflectance combination:

| Index | Ceiling | Walls | Reference plane |
|-------|---------|-------|-----------------|
| 0     | 0.7     | 0.5   | 0.2             |
| 1     | 0.7     | 0.3   | 0.2             |
| 2     | 0.5     | 0.5   | 0.2             |
| 3     | 0.5     | 0.3   | 0.2             |
| 4     | 0.3     | 0.3   | 0.2             |

```python
# UGR for room 4H×8H (row 10), crosswise, reflectances 70/50/20 (col 0)
ugr = result.values[10, 0]
print(f"UGR = {ugr:.1f}")
```

## Export to CSV

```python
csv_string = result.to_csv()
print(csv_string)
# 9.6,11.1,9.9,11.5,11.8,11.0,12.6,11.4,12.9,13.2
# 10.8,12.2,11.2,12.5,12.9,12.6,14.0,12.9,14.3,14.7
# ...

with open("ugr_table.csv", "w") as f:
    f.write(csv_string)
```

## Room configuration index

| Row | X/H  | Y/H  | k    |
|-----|------|------|------|
| 0   | 2    | 2    | 1.00 |
| 1   | 2    | 3    | 1.20 |
| 2   | 2    | 4    | 1.33 |
| 3   | 2    | 6    | 1.50 |
| 4   | 2    | 8    | 1.60 |
| 5   | 2    | 12   | 1.71 |
| 6   | 4    | 2    | 1.33 |
| 7   | 4    | 3    | 1.71 |
| 8   | 4    | 4    | 2.00 |
| 9   | 4    | 6    | 2.40 |
| 10  | 4    | 8    | 2.67 |
| 11  | 4    | 12   | 3.00 |
| 12  | 8    | 4    | 2.67 |
| 13  | 8    | 6    | 3.43 |
| 14  | 8    | 8    | 4.00 |
| 15  | 8    | 12   | 4.80 |
| 16  | 12   | 4    | 3.00 |
| 17  | 12   | 6    | 4.00 |
| 18  | 12   | 8    | 4.80 |

H = 2.0 m (luminaire height above observer eye level, CIE 190:2010 §4.2).
