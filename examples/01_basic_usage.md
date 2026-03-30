# Basic usage

> ⚠️ This example will be completed once `eulumdat-ugr` reaches v1.0.0.

```python
from pyldt import LdtReader
from eulumdat_ugr import UgrCalculator

ldt = LdtReader.read("luminaire.ldt")
result = UgrCalculator.compute(ldt)

print(result)
# UgrResult(ugr_4x8_longitudinal=18.3, ugr_4x8_transversal=18.1,
#           ugr_8x4_longitudinal=18.3, ugr_8x4_transversal=18.1)

for row in result.table():
    print(row)
# {"config": "4Hx8H", "direction": "longitudinal", "refl": "70/50/20", "ugr": 18.3}
# {"config": "4Hx8H", "direction": "transversal",  "refl": "70/50/20", "ugr": 18.1}
# {"config": "8Hx4H", "direction": "longitudinal", "refl": "70/50/20", "ugr": 18.3}
# {"config": "8Hx4H", "direction": "transversal",  "refl": "70/50/20", "ugr": 18.1}
```
