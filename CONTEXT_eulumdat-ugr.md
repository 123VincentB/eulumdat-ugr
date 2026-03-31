# CONTEXT — eulumdat-ugr

## Statut
- Package : **en cours — étapes 0, 1, 2, 3, 4, 5 complétées**
- Repo GitHub prévu : https://github.com/123VincentB/eulumdat-ugr
- PyPI : stub réservé
- Environnement virtuel : `eulumdat-ugr/.venv/`

---

## Description
Moteur Python de calcul UGR (Unified Glare Rating) conforme aux normes **CIE 117-1995**
et **CIE 190:2010**. Produit les 4 valeurs UGR catalogue pour un luminaire EULUMDAT,
en mode **corrected** (flux réel, non normalisé à 1000 lm).

---

## Écosystème eulumdat-*
| Package              | Rôle                               | Statut     | Import         |
|----------------------|------------------------------------|------------|----------------|
| `eulumdat-py`        | Lecture/écriture LDT               | v1.0.0 ✓  | `pyldt`        |
| `eulumdat-symmetry`  | Symétrisation, auto-détection ISYM | v1.0.0 ✓  | `ldt_symmetry` |
| `eulumdat-plot`      | Diagrammes polaires d'intensité    | v1.0.2 ✓  | `eulumdat_plot`|
| `eulumdat-luminance` | Calcul luminances, interpolation   | v1.1.1 ✓  | `eulumdat_luminance` |
| `eulumdat-ugr`       | Calcul UGR (CIE 117/190)           | non débuté | `eulumdat_ugr` |

Dépendances directes : `eulumdat-py >= 1.0.0`, `eulumdat-luminance >= 1.2.0`

---

## Périmètre fonctionnel

### Sorties attendues — les 4 valeurs UGR catalogue

| Configuration salle | Direction observation | Réflectances C/W/R | SHR  |
|---------------------|-----------------------|--------------------|------|
| 4H × 8H             | longitudinale (∥ Y)   | 70/50/20           | 0,25 |
| 4H × 8H             | transversale (⊥ Y)    | 70/50/20           | 0,25 |
| 8H × 4H             | longitudinale (∥ Y)   | 70/50/20           | 0,25 |
| 8H × 4H             | transversale (⊥ Y)    | 70/50/20           | 0,25 |

*(les 4 autres combinaisons de réflectances sont différées à v1.1.0)*

- **X** = dimension perpendiculaire à la ligne de visée
- **Y** = dimension parallèle à la ligne de visée
- Observation longitudinale : regard dans la direction du grand axe (Y=8H)
- Observation transversale : regard dans la direction du petit axe (Y=4H)

---

## Mode de calcul

### Stratégie uncorrected → corrected (CIE 190, §4.1, eq. 2)

Le calcul est effectué en deux étapes :

**Étape 1 — UGR uncorrected** (Φ₀ = 1000 lm normalisé)
```
UGR(Φ₀) = 8·log10( (0,25 / Lb₀) · Σ (L²₀_i · ω_i / p²_i) )
```
L₀_i et Lb₀ sont calculés avec Φ₀ = 1000 lm.

**Étape 2 — Correction flux réel** (CIE 190, §4.1, eq. 2)
```
UGR(Φ) = UGR(Φ₀) + 8·log10(Φ / Φ₀)
```
où `Φ = num_lamps[0] × lamp_flux[0]` (lm, premier set EULUMDAT).

Mathématiquement équivalent au calcul direct en flux réel (propriété du
logarithme). Avantage architectural : permet de valider UGR(Φ₀) indépendamment
contre les exemples CIE 190 (Table 1) avant d'appliquer la correction.

**Note importante :** la correction eq. 2 absorbe simultanément l'effet sur
L²_i (numérateur, proportionnel à Φ²) et sur Lb₀ (dénominateur, proportionnel
à Φ). Le rapport L²/Lb est donc proportionnel à Φ, ce qui donne le facteur
8·log10(Φ/Φ₀) global. Cette propriété doit être vérifiée en test unitaire.

### SHR — paramètre interne de validation

SHR est un **paramètre interne** de `UgrGrid`, non exposé dans l'API publique.

| SHR  | Usage | S (m) | N (4H×8H) |
|------|-------|--------|-----------|
| 0,25 | **Production** — valeurs catalogue fabricant | 0,5 | 512 (16×32) |
| 1,00 | **Validation uniquement** — reproduire exemples CIE 190 | 2,0 | 32 (4×8) ✓ Table 3 |

```python
# API publique — SHR=0,25 par défaut
result = UgrCalculator.compute(ldt)

# Validation interne uniquement — ne pas exposer dans l'API publique
result = UgrCalculator.compute(ldt, _shr=1.0)
```

**Avertissement traçabilité (ISO 17025) :** les Tables 4 et 5 de CIE 190 sont
définies pour SHR=1. Leur réutilisation pour SHR=0,25 est une convention
industrielle (DIALux/Relux), non formellement justifiée par la norme. Cette
hypothèse doit être documentée explicitement dans tout rapport d'audit.

---

## Références normatives
- **CIE 117-1995** : formule UGR, indice de position de Guth (Table 4.1), angle solide
- **CIE 190:2010** : méthode tabulaire, conditions standard, calcul de Lb (E_WID)

---

## Conventions géométriques (CIE 117, Figure 4.2)

Repère centré sur l'observateur :
- **R** : distance horizontale projetée sur la ligne de visée (luminaire → observateur, axe Y)
- **T** : distance horizontale perpendiculaire à la ligne de visée (axe X), valeur absolue
- **H** : hauteur du luminaire au-dessus de l'œil de l'observateur = **2,0 m** (fixe)

```
r² = R² + T² + H²                                    (CIE 117, eq. 4.8)
γ  = arctan(√(R² + T²) / H)                          (CIE 117, eq. 4.7)

# Luminaire crosswise (axe long ⊥ ligne de visée) :
C  = arctan(T / R)                                    (CIE 117, eq. 4.5)

# Luminaire endwise (axe long ∥ ligne de visée) :
C  = 90° - arctan(T / R)                              (CIE 117, eq. 4.6)
```

**Convention EULUMDAT :** γ=0° = nadir, C = azimut autour de l'axe vertical.
Les angles (C, γ) calculés ci-dessus sont directement utilisables dans `result.at()`.

---

## Géométrie standard (CIE 190:2010, §4.2)

- H = 2,0 m (hauteur luminaires au-dessus du plan de référence = niveau yeux)
- Plan de référence = hauteur yeux observateur = 1,2 m du sol
- Hauteur luminaires = 3,2 m du sol
- Observateur : milieu du mur court, sur le mur (y_R = 0), centré sur l'axe X (x_T = 0)
- Regard horizontal

### Grille de luminaires (SHR = 0,25)
```
S = SHR × H = 0,25 × 2,0 = 0,5 m
```

**Positions des luminaires :**
- Axe X (T) : ±S/2, ±3S/2, ±5S/2, ... → symétrique par rapport à x_T = 0
- Axe Y (R) : S/2, 3S/2, 5S/2, ... → à partir du mur observateur

**Nombre de luminaires (SHR=0,25) :**

| Salle   | Dimensions réelles | Nx total | Ny total | N total |
|---------|-------------------|----------|----------|---------|
| 4H×8H   | 8 m × 16 m        | 16       | 32       | 512     |
| 8H×4H   | 16 m × 8 m        | 32       | 16       | 512     |

**Note :** CIE 190 Table 3 donne N=32 pour SHR=1,0 (4×8 positions). Pour SHR=0,25, N=512.

Seuls les luminaires **à l'intérieur du local** sont inclus dans la somme.
Luminaires avec T/R > 3 ou γ > 85° : **ignorés** (recommandation CIE 117).

### Room index k
```
k = (X × Y) / (H × (X + Y))
```
- 4H×8H : k = (8 × 16) / (2 × 24) = **2,67**
- 8H×4H : k = (16 × 8) / (2 × 24) = **2,67** (identique)

---

## Formule UGR (CIE 117, eq. 1 / CIE 190, eq. 1)

```
UGR = 8 · log10( (0,25 / Lb) · Σ (L²_i · ω_i / p²_i) )
```

### L_i — luminance apparente du luminaire i (cd/m²)
Obtenue par interpolation bilinéaire depuis `eulumdat-luminance` :
```python
from eulumdat_luminance import LuminanceCalculator
result = LuminanceCalculator.compute(ldt, full=True)
L_i = result.at(c_deg=C_i, g_deg=gamma_i)   # flux réel, cd/m²
```
**Mode corrected** : flux réel du luminaire, `conv_factor = 1` toujours dans EULUMDAT.

### ω_i — angle solide apparent (sr) (CIE 117, eq. 4.4)
```
ω_i = A_p(C_i, γ_i) / r²_i
```
- `A_p` = aire projetée de la surface lumineuse vue depuis (C_i, γ_i), en m²
- Même calcul que dans `eulumdat-luminance` (`LuminanceCalculator`)
- `r²_i = R²_i + T²_i + H²`

### p_i — indice de position de Guth
Interpolation bilinéaire dans **Table 4.1 de CIE 117** sur les paramètres :
- `H/R` (ligne) et `T/R` (colonne), valeurs absolues
- p est symétrique en T/R
- Luminaires hors table (T/R > 3 ou H/R hors plage) : **ignorés**

La Table 4.1 est encodée en dur dans le module comme constante numpy.

---

## Calcul de Lb (luminance de fond)

```
Lb = E_WID / π
```

### E_WID — illuminance indirecte verticale à l'œil (CIE 190, eq. 7/8)
```
E_WID = B · F_UWID

B = Φ_réel · N / A_w                    # mode corrected (remplace 1000·N/A_w)
```
où :
- `Φ_réel = num_lamps[0] × lamp_flux[0]` (lm, premier set EULUMDAT)
- `N` = nombre de luminaires dans le local
- `A_w` = surface totale des murs entre plan de référence et plan des luminaires (m²)
  ```
  A_w = 2 · H · (X + Y)
  ```

### F_UWID (CIE 190, eq. 8b)
```
F_UWID = F_DF · F_T,FW + F_DW · F_T,WW-1 + F_DC · F_T,CW
```

#### Facteurs de distribution flux (calculés depuis I(C,γ) du LDT)

Flux zonaux calculés aux **milieux de zones de 10°** (CIE 190 §4.2) :
```python
# Milieux : γ_mid = 5°, 15°, 25°, …, 175°  (18 zones sur la sphère complète)
# Les données aux angles 0°, 10°, 20°, … sont informationnelles uniquement.

# Zone Factor (approximation du milieu de zone) :
ZF(γ_mid) = 2π · sin(γ_mid_rad) · Δγ_rad    (Δγ = 10°)

# Intensité moyenne au milieu de zone (interpolation linéaire + moyenne sur C) :
I_avg(γ_mid) = moyenne de I(C, γ_mid) sur tous les plans C

# Flux zonal nominal (base 1000 lm) :
G(γ_mid) = I_avg(γ_mid) × ZF(γ_mid)

# Normalisation par LORL :
G_H = G × (LORL/100) / (Σ G / 1000)

R_DLO = Σ G_H(γ_mid = 5°..85°)  / 1000   # Downward Light Output Ratio
R_ULO = Σ G_H(γ_mid = 95°..175°) / 1000   # Upward Light Output Ratio
```

Flux zonaux cumulés (indices sur les 9 zones descendantes 5°…85°) :
```
Φ_zL1 = G_H[:4].sum() + 0,130 × G_H[4]   # C(40°) + 0,130 × G_H(45°)
Φ_zL2 = G_H[:6].sum()                      # C(60°)
Φ_zL3 = G_H[:7].sum() + 0,547 × G_H[7]   # C(70°) + 0,547 × G_H(75°)
Φ_zL4 = G_H.sum()                          # C(90°)
```

Flux indirect vers les surfaces :
```
Φ_zL = Φ_zL1·F_GL1 + Φ_zL2·F_GL2 + Φ_zL3·F_GL3 + Φ_zL4·F_GL4   (eq. 9)

F_DF  = Φ_zL / Φ_0 = Φ_zL / 1000     (eq. 10)
F_DW  = R_DLO - F_DF                   (eq. 11)
F_DC  = R_ULO                          (eq. 12)
```

#### Facteurs géométriques F_GL (Table 4 CIE 190)
Pour k = 2,67 (4H×8H et 8H×4H) — encodés en dur pour v1.0.0 :

| F_GL1 | F_GL2 | F_GL3 | F_GL4 |
|-------|-------|-------|-------|
| 0,280 | 0,165 | 0,499 | 0,006 |

#### Facteurs de transfert F_T (Table 5 CIE 190)
Encodés en dur pour k = 2,67 et les **5 combinaisons de réflectances** :

| Réflectances C/W/R | F_T,FW | F_T,WW-1 | F_T,CW |
|--------------------|--------|----------|--------|
| 70/50/20           | 0,115  | 0,211    | 0,307  |
| 70/30/20           | 0,106  | 0,117    | 0,283  |
| 50/50/20           | 0,100  | 0,187    | 0,211  |
| 50/30/20           | 0,093  | 0,104    | 0,196  |
| 30/30/20           | 0,081  | 0,092    | 0,114  |

---

## Structure du projet

```
eulumdat-ugr/
├── data/
│   ├── input/              # fichiers LDT de test (sample_01.ldt … sample_11.ldt)
│   │                       # sample_11.ldt = luminaire exemple CIE 190:2010 (référence principale)
│   ├── output/             # résultats (ignoré par git)
│   └── reference/
│       └── relux_reference.json   # valeurs UGR de référence DIALux/Relux
├── docs/                   # tableaux normatifs (ignoré par git)
│   ├── cie190-table2.csv   # géométrie : xT/H vs yR/H → C°, γ°, K, H/D, yR/D, xT/D
│   ├── cie190-table3.csv   # configurations salles : X/Y dim, k, N, Aw, B
│   ├── cie190-table4.csv   # facteurs F_GL partiels (10 room sizes)
│   ├── cie190-table5.csv   # facteurs F_T complets : 19 salles × 5 réflectances × 3 facteurs
│   └── cie190-sample.csv   # matrice intensités 37γ × 24C (cd/klm) — sample_11
├── examples/
│   └── 01_basic_usage.md
├── src/
│   └── eulumdat_ugr/
│       ├── __init__.py
│       ├── geometry.py     # UgrGrid, grille luminaires, vecteurs R/T/H → C/γ/r
│       ├── guth.py         # GuthTable, Table 4.1 CIE 117, interpolation p
│       ├── photometry.py   # UgrPhotometry, L_i, ω_i depuis LuminanceResult
│       ├── background.py   # BackgroundLuminance, flux zonaux, F_UWID, Lb
│       └── ugr.py          # UgrCalculator, assemblage final, 4 valeurs
├── tests/
│   └── test_ugr.py
├── .gitignore
├── CLAUDE.md
├── CONTEXT_eulumdat-ugr.md
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Données de référence disponibles

### Fichiers LDT (`data/input/`)
- `sample_01.ldt` à `sample_10.ldt` : luminaires de test variés
- `sample_11.ldt` : **luminaire exemple CIE 190:2010** — référence principale pour la validation numérique

### Tableaux CIE 190 (`docs/`)

| Fichier | Contenu | Utilisation |
|---------|---------|-------------|
| `cie190-table2.csv` | Géométrie : xT/H vs yR/H → C°, γ°, K, H/D, yR/D, xT/D | Référence géométrique |
| `cie190-table3.csv` | 19 configurations salles : X/Y, k, N, Aw, B | Validation configurations |
| `cie190-table4.csv` | F_GL partiels pour 10 tailles de salle | Facteurs géométriques |
| `cie190-table5.csv` | F_T complets : 19 salles × **5 réflectances** × 3 facteurs (FW/WW-1/CW) | Calcul Lb |
| `cie190-table8.csv` | Worksheet flux zonaux sample_11 (milieux 10°, 24 plans C, G, G_H) | Validation `background.py` |
| `cie190-2010_page20-22.docx` | Exemple CIE 190 pour salle 2H×4H (k=1,33) : algorithme complet, F_DF/F_DW/Lb attendus | Validation algorithme |

**Valeurs de référence validées (sample_11, SHR=1, 4H×8H, k=2,67) :**
- R_DLO ≈ 0,6497 ; R_ULO ≈ 0,0000
- F_DF ≈ 0,4968 ; F_DW ≈ 0,1529 ; F_DC ≈ 0,0000
- F_UWID (70/50/20) ≈ 0,08942
- **Lb (70/50/20) ≈ 9,49 cd/m²**

**Validation croisée k=1,33 (docx) :** F_DF=0,390 ✓, F_DW=0,260 ✓ (correspondance exacte).

---

## API publique (cible)

```python
from pyldt import LdtReader
from eulumdat_ugr import UgrCalculator

ldt = LdtReader.read("file.ldt")
result = UgrCalculator.compute(ldt)

# Accès aux 4 valeurs UGR (dict réflectances → float)
result.ugr_4x8_longitudinal   # {"70/50/20": 18.3, "70/30/20": 19.1, ...}
result.ugr_4x8_transversal
result.ugr_8x4_longitudinal
result.ugr_8x4_transversal

# Liste de dicts — pour validation et export
result.table()
# [
#   {"config": "4Hx8H", "direction": "longitudinal", "refl": "70/50/20", "ugr": 18.3},
#   {"config": "4Hx8H", "direction": "transversal",  "refl": "70/50/20", "ugr": 18.1},
#   {"config": "8Hx4H", "direction": "longitudinal", "refl": "70/50/20", "ugr": 18.3},
#   {"config": "8Hx4H", "direction": "transversal",  "refl": "70/50/20", "ugr": 18.1},
# ]  # 4 entrées en v1.0.0

# Résumé texte
print(result)
```

### Prérequis : eulumdat-luminance v1.2.0
Avant d'implémenter `photometry.py`, ajouter dans `eulumdat-luminance` :
```python
# Dans LuminanceResult (result.py)
def projected_area(self, c_deg: float, g_deg: float) -> float:
    """Aire projetée de la surface lumineuse (m²) vue depuis (C, γ)."""
    ...
```
Cette méthode expose le calcul A_proj déjà présent dans `LuminanceCalculator`.

---

## Architecture des modules

### `geometry.py` — UgrGrid
- Génère la grille de luminaires pour une configuration (X_dim, Y_dim, S=0,5m)
- Calcule (R_i, T_i) pour chaque luminaire depuis l'observateur
- Calcule (C_i, γ_i, r_i) selon orientation (crosswise / endwise) et eq. 4.5–4.8
- Filtre : T/R > 3 ou γ > 85° → ignoré

### `guth.py` — GuthTable
- Table 4.1 CIE 117 encodée en dur (numpy array)
- Plage : T/R = 0,00..3,00, H/R = 0,00..3,00 (limite effective de la table)
- `p(H_R, T_R)` : interpolation bilinéaire, entrées = valeurs absolues
- Retourne NaN si hors plage → luminaire ignoré dans la somme

### `photometry.py` — UgrPhotometry
- Prend un `LuminanceResult` (eulumdat-luminance) et la géométrie d'un luminaire
- Calcule L_i = `result.at(C_i, γ_i)`
- Calcule A_p(C_i, γ_i) via `result.projected_area(C_i, γ_i)` — méthode ajoutée
  dans eulumdat-luminance v1.2.0 (à publier avant eulumdat-ugr)
- Calcule ω_i = A_p / r²

### `background.py` — BackgroundLuminance
- Calcule les flux zonaux depuis I(C,γ) du LDT (intégration numérique)
- Calcule R_LO, R_DLO, R_ULO
- Calcule Φ_zL1..4 et Φ_zL (eq. 9)
- Calcule F_DF, F_DW, F_DC (eq. 10–12)
- Calcule F_UWID pour chaque combinaison réflectances (Table 5 encodée)
- Calcule B = Φ_réel · N / A_w (mode corrected)
- Retourne Lb pour chaque combinaison réflectances

### `ugr.py` — UgrCalculator
- Point d'entrée principal
- Orchestre geometry + photometry + guth + background
- Produit les 4 configurations × 5 réflectances = **20 valeurs UGR**

---

## Dépendances pyproject.toml

```toml
[project]
name = "eulumdat-ugr"
dependencies = [
    "eulumdat-py >= 1.0.0",
    "eulumdat-luminance >= 1.2.0",
    "numpy >= 1.24.0",
    "scipy >= 1.10.0",
]
```

---

## Hypothèses verrouillées

| Hypothèse | Valeur | Source |
|-----------|--------|--------|
| H | 2,0 m | CIE 190 §4.2 |
| SHR | 0,25 | Convention industrielle (DIALux/Relux) |
| S | 0,5 m | S = SHR × H |
| Hauteur yeux | 1,2 m du sol | CIE 190 §4.2 |
| Hauteur luminaires | 3,2 m du sol | CIE 190 §4.2 |
| Position observateur | Milieu mur, sur le mur (y_R=0, x_T=0) | CIE 190 §4.2 |
| Luminaires | Intérieur du local uniquement | |
| Filtre T/R | > 3 → ignoré | CIE 117 §4.5 |
| Filtre γ | > 85° → ignoré | CIE 117 |
| conv_factor | 1 (toujours) | EULUMDAT projet |
| Flux | Premier set uniquement | Cohérence eulumdat-luminance |
| Mode | corrected (flux réel) | Objectif projet |
| Φ_bare_lamp (ratios) | 1000 lm | CIE 190 (normalisé pour R_LO) |
| k pour Table 4/5 | 2,67 (les deux salles) | Calculé |
| F_GL | Valeurs k=2,67 Table 4 CIE 190 | Encodées en dur |

---

## Stratégie de validation

1. **Validation unitaire** : reproduire l'exemple CIE 190 (Table 7/8) — vérifier R_LO, R_DLO, Φ_zL, F_UWID
2. **Validation numérique** : comparer les 4 valeurs UGR (70/50/20) contre DIALux/Relux sur 5–10 fichiers LDT de référence
3. **Tolérance cible** : ±0,5 UGR (arrondi standard des tables catalogue)

---

## Historique des versions

| Version | Date | Changements |
|---------|------|-------------|
| 0.0.1 | 2026-03 | Squelette projet + guth.py (Table 4.1 CIE 117, 25 tests) |
| 0.0.2 | 2026-03 | geometry.py — UgrGrid, grille luminaires, filtres T/R et γ (25 tests) |
| 0.0.3 | 2026-03 | background.py — flux zonaux 10° milieux, LORL, Lb × 5 réflectances (13 tests) |
| 0.0.4 | 2026-03 | photometry.py — L_i, ω_i vectorisés depuis LuminanceResult (12 tests) |

---

## Ordre d'implémentation (Claude Code)

**Une étape à la fois. Attendre validation avant de passer à la suivante.**

### Étape 0 ✓ — eulumdat-luminance v1.2.0 (repo eulumdat-luminance)
- ✓ `LuminanceResult.projected_area(c_deg, g_deg)` ajoutée, 10 tests, publié PyPI

### Étape 1 ✓ — Squelette du projet eulumdat-ugr
- ✓ Structure de dossiers, `pyproject.toml`, `__init__.py`, `.gitignore`, `CLAUDE.md`

### Étape 2 ✓ — `guth.py`
- ✓ Table 4.1 CIE 117 (31×20, NaN pour cellules manquantes), interpolation bilinéaire
- ✓ 25 tests passants

### Étape 3 ✓ — `geometry.py`
- ✓ `UgrGrid(x_dim_H, y_dim_H, shr=0.25)` : génère la grille de luminaires
- ✓ Calcul (R_i, T_i) depuis l'observateur pour chaque luminaire
- ✓ Calcul (C_i, γ_i, r_i) selon orientation (crosswise / endwise) et eq. 4.5–4.8 CIE 117
- ✓ Filtres : T/R > 3 ou γ > 85° → luminaire ignoré
- ✓ 25 tests passants

### Étape 4 ✓ — `background.py`
- ✓ Flux zonaux aux milieux de zones 10° (5°…175°), interpolation + moyenne sur plans C
- ✓ Normalisation par LORL → G_H, R_DLO, R_ULO
- ✓ Φ_zL1…4 par indices fixes sur les 9 zones descendantes (eq. 9 CIE 190)
- ✓ F_DF, F_DW, F_DC (eq. 10–12) ; F_UWID et Lb pour les 5 combinaisons réflectances
- ✓ Validé contre Table 8 CIE 190 (k=1,33 : F_DF=0,390 ✓, F_DW=0,260 ✓)
- ✓ 13 tests passants

### Étape 5 ✓ — `photometry.py`
- ✓ `UgrPhotometry.compute(lum_result, C_deg, gamma_deg, r_m)` → `(L, omega)`
- ✓ L_i = `result.at(C_i, γ_i)`, ω_i = `result.projected_area(C_i, γ_i)` / r²_i
- ✓ Entièrement vectorisé (numpy), `full=True` requis
- ✓ 12 tests passants

### Étape 6 — `ugr.py`
- `UgrCalculator.compute(ldt, _shr=0.25)` : point d'entrée principal
- Orchestre geometry + photometry + guth + background
- Calcul Σ L²_i · ω_i / p²_i
- UGR uncorrected = 8·log10(0.25/Lb₀ · Σ)
- Correction flux réel : UGR(Φ) = UGR(Φ₀) + 8·log10(Φ/Φ₀)
- `UgrResult.table()` → liste de 4 dicts

### Étape 7 — Validation finale
- Tester SHR=1.0 → comparer contre exemples CIE 190
- Tester SHR=0.25 → comparer contre valeurs DIALux/Relux (5–10 fichiers LDT fournis)
- Tolérance cible : ±0,5 UGR

---

## Roadmap

- **v1.0.0** : 4 configurations × réflectances **70/50/20** = **4 valeurs UGR**, mode corrected, validation DIALux/Relux
- **v1.1.0** : 5 combinaisons de réflectances complètes → 20 valeurs UGR
- **v1.2.0** : configurations supplémentaires (2H×2H ... 12H×12H), SHR paramétrable
