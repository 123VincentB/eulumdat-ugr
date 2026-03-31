# -*- coding: utf-8 -*-
"""
ugr.py
------
UgrCalculator — computes the full 19×10 UGR table for an EULUMDAT luminaire.

Table structure
---------------
- 19 rows : standard room configurations (CIE 190:2010)
- 10 columns :
    cols 0–4  → crosswise orientation, 5 reflectance combinations
    cols 5–9  → endwise orientation,   5 reflectance combinations
  col % 5 → reflectance index: 0=70/50/20, 1=70/30/20, 2=50/50/20,
                                 3=50/30/20, 4=30/30/20

UGR formula (CIE 117:1995 eq. 1)
----------------------------------
    UGR = 8 · log10( (0.25 / Lb) · Σ (L²_i · ω_i / p²_i) )

Room geometry (CIE 190:2010 §4.2)
-----------------------------------
    H = 2.0 m  (luminaire height above observer eye level)
    A_w = 2 · H · (X + Y)   [m²]
    SHR = 0.25  (default catalogue value)

Background luminance Lb (CIE 190:2010 §4.2)
--------------------------------------------
Lb depends only on the room geometry (N, A_w, room_idx) and the luminaire's
photometric distribution.  It is identical for crosswise and endwise observation.
"""

import numpy as np
from eulumdat_luminance import LuminanceCalculator

from .background import BackgroundLuminance, REFLECTANCES, _ROOM_CONFIGS
from .geometry import UgrGrid, H_MOUNT
from .guth import GuthTable
from .photometry import UgrPhotometry

_N_ROOMS: int = 19
_N_COLS: int = 10
_ORIENTATIONS: tuple[str, str] = ("crosswise", "endwise")


class UgrResult:
    """
    Full 19×10 UGR table for one luminaire.

    Attributes
    ----------
    values : np.ndarray, shape (19, 10)
        UGR values.  Rows = room configs, cols 0–4 = crosswise,
        cols 5–9 = endwise.  np.nan where UGR could not be computed
        (e.g. no valid luminaire positions or Lb ≤ 0).
    """

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def to_csv(self, fmt: str = "{:.1f}") -> str:
        """
        Return the 19×10 table as a comma-separated string (no header).

        Parameters
        ----------
        fmt : str
            Python format string for each value.  Default ``"{:.1f}"``.
        """
        lines = []
        for row in self.values:
            lines.append(",".join(fmt.format(v) for v in row))
        return "\n".join(lines)


class UgrCalculator:
    """
    Compute the full 19×10 UGR table for an EULUMDAT luminaire.

    Usage
    -----
    ::

        from pyldt import LdtReader
        from eulumdat_ugr import UgrCalculator

        ldt = LdtReader.read("luminaire.ldt")
        result = UgrCalculator.compute(ldt)
        print(result.to_csv())
    """

    @classmethod
    def compute(cls, ldt, _shr: float = 0.25) -> UgrResult:
        """
        Compute the full UGR table.

        Parameters
        ----------
        ldt : pyldt.model.Ldt
            EULUMDAT luminaire data.
        _shr : float, optional
            Spacing-to-Height Ratio (default 0.25 = catalogue values).
            Use 1.0 to reproduce CIE 190:2010 numerical validation examples.

        Returns
        -------
        UgrResult
            19×10 UGR matrix.
        """
        # Luminance result — computed once, reused for all rooms / orientations
        lum_result = LuminanceCalculator.compute(ldt, full=True)

        values = np.full((_N_ROOMS, _N_COLS), np.nan)

        for room_idx, (x_dim_H, y_dim_H) in enumerate(_ROOM_CONFIGS):
            grid = UgrGrid(x_dim_H, y_dim_H, shr=_shr)
            n_lum = grid.n_total
            # A_w = 2 · H · (X + Y),  X = x_dim_H·H,  Y = y_dim_H·H,  H = 2 m
            # → A_w = 2 · H² · (x_dim_H + y_dim_H) = 8 · (x_dim_H + y_dim_H)
            a_w = 8.0 * (x_dim_H + y_dim_H)

            # Lb is the same for crosswise and endwise (same room, same LDT)
            lb_map = BackgroundLuminance.compute(ldt, n_lum, a_w, room_idx=room_idx)

            for orient_idx, orientation in enumerate(_ORIENTATIONS):
                R, T, C_deg, gamma_deg, r_m = grid.angles(orientation)
                if len(R) == 0:
                    continue

                L, omega = UgrPhotometry.compute(lum_result, C_deg, gamma_deg, r_m)

                # Guth position index — vectorized
                h_r = H_MOUNT / R   # H/R where R is depth (line-of-sight distance)
                t_r = T / R         # T/R (T is already absolute from geometry.py)
                p = GuthTable.p_vec(h_r, t_r)

                # Valid luminaires: p not NaN, L > 0, omega > 0
                valid = ~np.isnan(p) & (L > 0) & (omega > 0)
                if not np.any(valid):
                    continue

                ugr_sum = float(np.sum(
                    L[valid] ** 2 * omega[valid] / p[valid] ** 2
                ))

                for refl_idx, refl in enumerate(REFLECTANCES):
                    lb = lb_map[refl]
                    if lb <= 0.0:
                        continue
                    col = orient_idx * 5 + refl_idx
                    values[room_idx, col] = 8.0 * np.log10(0.25 / lb * ugr_sum)

        return UgrResult(values)
